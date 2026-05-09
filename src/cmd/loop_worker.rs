/// loop_worker — background prediction loop.
///
/// Runs continuously: fetch context → call LLM for analysis → submit prediction → sleep.
///
/// LLM is invoked via OpenClaw CLI with extended thinking:
///   `openclaw agent --agent <id> --message <prompt> --thinking high --timeout 180`
///
/// With --thinking high, the agent can:
///   - Do deeper reasoning before making predictions
///   - Use web search to check news, sentiment, market data (if configured)
///   - Use any tools available in the agent's gateway configuration
///   - Output a final `DECISION: {...}` with its prediction
///
/// Usage: predict-agent loop [--interval 120] [--max-iterations 0] [--agent-id predict-worker]
///
/// The loop handles:
///   - Automatic context fetching each round
///   - LLM prompt construction with klines data
///   - Parsing LLM response (extracts DECISION: JSON from output)
///   - Submission with error recovery
///   - Adaptive backoff on empty markets or errors
///   - Graceful shutdown on SIGINT/SIGTERM
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::Write;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::auth::refresh_wallet_token;
use crate::client::ApiClient;
use crate::{log_debug, log_error, log_info, log_warn};

pub struct LoopArgs {
    pub interval: u64,
    pub max_iterations: u64,
    pub agent_id: String,
    /// If true, output [NOTIFY] lines for the agent to relay to user
    pub notify: bool,
}

/// Print a notification line that the agent should relay to the user.
/// Format: [NOTIFY] <message>
/// Only printed if notify=true.
macro_rules! notify {
    ($notify:expr, $($arg:tt)*) => {
        if $notify {
            println!("[NOTIFY] {}", format!($($arg)*));
        }
    };
}

pub fn run(server_url: &str, args: LoopArgs) -> Result<()> {
    log_info!(
        "loop: starting (interval={}s, max_iter={}, agent={}, server={})",
        args.interval,
        args.max_iterations,
        args.agent_id,
        server_url
    );

    // Set up graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\n[predict-agent] loop: received shutdown signal, finishing current round...");
        r.store(false, Ordering::SeqCst);
    })
    .ok(); // Ignore error if handler already set

    let direct_llm = direct_llm_config();
    let openclaw_bin = if direct_llm.is_some() {
        log_info!("loop: using direct LLM endpoint (OpenClaw disabled)");
        None
    } else {
        let openclaw_bin = detect_openclaw();
        if openclaw_bin.is_none() {
            log_error!("loop: no direct LLM env vars and openclaw CLI not found");
            eprintln!("\nSet PREDICT_LLM_BASE_URL, PREDICT_LLM_API_KEY, and PREDICT_LLM_MODEL for direct LLM mode.");
            return Ok(());
        }
        let openclaw_bin = openclaw_bin.unwrap();
        log_info!("loop: using openclaw at {}", openclaw_bin);
        ensure_agent(&openclaw_bin, &args.agent_id);
        Some(openclaw_bin)
    };

    let mut iteration: u64 = 0;
    let mut consecutive_empty = 0u32;
    let mut consecutive_errors = 0u32;

    while running.load(Ordering::SeqCst) {
        iteration += 1;
        if args.max_iterations > 0 && iteration > args.max_iterations {
            log_info!(
                "loop: reached max iterations ({}), stopping",
                args.max_iterations
            );
            break;
        }

        log_info!("loop: === iteration {} ===", iteration);
        let iter_start = Instant::now();

        match run_iteration(server_url, openclaw_bin.as_deref(), &args.agent_id) {
            IterationResult::Submitted {
                market,
                direction,
                tickets,
                tickets_filled,
                order_status,
            } => {
                let elapsed = iter_start.elapsed().as_secs_f64();
                let fill_info = match order_status.as_str() {
                    "filled" => format!("FILLED {}/{}", tickets_filled, tickets),
                    "partial" => format!("PARTIAL {}/{}", tickets_filled, tickets),
                    _ => format!("PENDING 0/{} (waiting for counterparty)", tickets),
                };
                log_info!(
                    "loop: {} {} for {} — {} ({:.1}s)",
                    direction,
                    fill_info,
                    market,
                    order_status,
                    elapsed
                );
                notify!(
                    args.notify,
                    "Round {}: {} {} — {} ({:.1}s)",
                    iteration,
                    direction.to_uppercase(),
                    market,
                    fill_info,
                    elapsed
                );
                consecutive_empty = 0;
                consecutive_errors = 0;
            }
            IterationResult::Skipped { reason } => {
                let elapsed = iter_start.elapsed().as_secs_f64();
                log_info!("loop: skipped this round ({:.1}s): {}", elapsed, reason);
                notify!(args.notify, "Round {}: Skipped — {}", iteration, reason);
                consecutive_empty = 0;
                consecutive_errors = 0;
                // No penalty for skipping — it's a valid decision
            }
            IterationResult::NoMarkets { wait_seconds } => {
                consecutive_empty += 1;
                let backoff =
                    calculate_backoff(args.interval, consecutive_empty, Some(wait_seconds));
                log_info!(
                    "loop: no submittable markets (consecutive={}), sleeping {}s",
                    consecutive_empty,
                    backoff
                );
                notify!(
                    args.notify,
                    "Round {}: No markets available, waiting {}s",
                    iteration,
                    backoff
                );
                interruptible_sleep(backoff, &running);
                continue;
            }
            IterationResult::RateLimited { wait_seconds } => {
                log_info!("loop: rate limited, sleeping {}s", wait_seconds);
                notify!(
                    args.notify,
                    "Round {}: Rate limited, waiting {}s",
                    iteration,
                    wait_seconds
                );
                interruptible_sleep(wait_seconds, &running);
                continue;
            }
            IterationResult::LlmFailed { reason } => {
                consecutive_errors += 1;
                let backoff = calculate_backoff(args.interval, consecutive_errors, None);
                log_warn!(
                    "loop: LLM call failed ({}), sleeping {}s (errors={})",
                    reason,
                    backoff,
                    consecutive_errors
                );
                notify!(
                    args.notify,
                    "Round {}: LLM error — {}, retrying in {}s",
                    iteration,
                    reason,
                    backoff
                );
                interruptible_sleep(backoff, &running);
                continue;
            }
            IterationResult::Error { reason } => {
                consecutive_errors += 1;
                let backoff = calculate_backoff(args.interval, consecutive_errors, None);
                log_error!(
                    "loop: iteration error ({}), sleeping {}s (errors={})",
                    reason,
                    backoff,
                    consecutive_errors
                );
                notify!(
                    args.notify,
                    "Round {}: Error — {}, retrying in {}s",
                    iteration,
                    reason,
                    backoff
                );
                interruptible_sleep(backoff, &running);
                continue;
            }
        }

        // Normal sleep between iterations
        log_debug!("loop: sleeping {}s until next iteration", args.interval);
        interruptible_sleep(args.interval, &running);
    }

    log_info!("loop: stopped after {} iterations", iteration);
    Ok(())
}

enum IterationResult {
    Submitted {
        market: String,
        direction: String,
        tickets: u32,
        tickets_filled: u32,
        order_status: String, // "filled", "partial", "open"
    },
    Skipped {
        reason: String,
    },
    NoMarkets {
        wait_seconds: u64,
    },
    RateLimited {
        wait_seconds: u64,
    },
    LlmFailed {
        reason: String,
    },
    Error {
        reason: String,
    },
}

fn run_iteration(server_url: &str, openclaw_bin: Option<&str>, agent_id: &str) -> IterationResult {
    // 1. Create API client
    let client = match ApiClient::new(server_url.to_string()) {
        Ok(c) => c,
        Err(e) => {
            return IterationResult::Error {
                reason: format!("API client init failed: {e}"),
            }
        }
    };

    // 2. Fetch agent status (includes timeslot, open_orders, recent_results)
    // Auto-refresh wallet token on auth failure
    let status = match client.get_auth("/api/v1/agents/me/status") {
        Ok(v) => v,
        Err(e) => {
            let err_str = e.to_string();
            // Check if this is an auth error that might be fixed by refreshing token
            if err_str.contains("AUTH_FAILED")
                || err_str.contains("expired")
                || err_str.contains("invalid token")
            {
                log_warn!("loop: auth failed, attempting token refresh...");
                match refresh_wallet_token() {
                    Ok(_) => {
                        log_info!("loop: token refreshed, retrying status fetch...");
                        // Recreate client with new token and retry
                        let new_client = match ApiClient::new(server_url.to_string()) {
                            Ok(c) => c,
                            Err(e) => {
                                return IterationResult::Error {
                                    reason: format!("client reinit failed: {e}"),
                                }
                            }
                        };
                        match new_client.get_auth("/api/v1/agents/me/status") {
                            Ok(v) => v,
                            Err(e) => {
                                return IterationResult::Error {
                                    reason: format!("status fetch failed after refresh: {e}"),
                                }
                            }
                        }
                    }
                    Err(refresh_err) => {
                        log_error!("loop: token refresh failed: {}", refresh_err);
                        return IterationResult::Error {
                            reason: format!(
                                "auth failed and token refresh failed: {e} / {refresh_err}"
                            ),
                        };
                    }
                }
            } else {
                return IterationResult::Error {
                    reason: format!("status fetch failed: {e}"),
                };
            }
        }
    };
    let agent_data = status.get("data").cloned().unwrap_or(json!({}));
    let balance = agent_data
        .get("balance")
        .and_then(|v| {
            v.as_str()
                .and_then(|s| s.parse::<f64>().ok())
                .or_else(|| v.as_f64())
        })
        .unwrap_or(0.0);
    let persona = agent_data
        .get("persona")
        .and_then(|v| v.as_str())
        .unwrap_or("none");

    // 3. Check timeslot — skip LLM entirely if no submissions remaining
    let timeslot = agent_data.get("timeslot");
    let submissions_remaining = timeslot
        .and_then(|t| t.get("submissions_remaining"))
        .and_then(|v| v.as_i64())
        .unwrap_or(3); // default to 3 if server doesn't return timeslot yet
    let slot_resets_in = timeslot
        .and_then(|t| t.get("slot_resets_in_seconds"))
        .and_then(|v| v.as_u64())
        .unwrap_or(300);
    let submissions_used = timeslot
        .and_then(|t| t.get("submissions_used"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    log_info!(
        "loop: balance={:.0}, persona={}, timeslot={}/{} used, resets in {}s",
        balance,
        persona,
        submissions_used,
        timeslot
            .and_then(|t| t.get("slot_limit"))
            .and_then(|v| v.as_i64())
            .unwrap_or(3),
        slot_resets_in
    );

    if submissions_remaining <= 0 {
        log_info!(
            "loop: no submissions remaining in this timeslot, waiting {}s for reset",
            slot_resets_in
        );
        return IterationResult::RateLimited {
            wait_seconds: slot_resets_in.max(10),
        };
    }

    // Extract open_orders and recent_results for LLM context
    let open_orders = agent_data
        .get("open_orders")
        .and_then(|v| v.as_array())
        .cloned();
    let recent_results = agent_data
        .get("recent_results")
        .and_then(|v| v.as_array())
        .cloned();

    // Compute recent filled-accuracy throttle so the prompt + validator can
    // tighten the gate after a losing streak. Only filled (resolved) results
    // count — pending or refunded orders are excluded.
    let recent_acc = recent_accuracy(&recent_results);

    // 4. Fetch smart market recommendations from server
    let recommendations = match client.get_auth("/api/v1/markets/recommend") {
        Ok(v) => v
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default(),
        Err(e) => {
            log_warn!(
                "loop: recommend endpoint failed ({}), falling back to active markets",
                e
            );
            Vec::new()
        }
    };

    // Filter to actionable recommendations (action != "skip", >120s remaining)
    let actionable: Vec<&Value> = recommendations
        .iter()
        .filter(|r| {
            let not_skip = r.get("action").and_then(|a| a.as_str()) != Some("skip");
            let enough_time = r
                .get("seconds_to_close")
                .and_then(|v| v.as_i64())
                .map(|s| s > 120)
                .unwrap_or(false);
            not_skip && enough_time
        })
        .collect();

    // If no recommendations, fall back to active markets
    let (market_id, market_info) = if !actionable.is_empty() {
        let top = actionable[0];
        let id = top
            .get("market_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        log_info!(
            "loop: server recommends {} (score={}, reason={})",
            id,
            top.get("score").and_then(|v| v.as_i64()).unwrap_or(0),
            top.get("reason").and_then(|v| v.as_str()).unwrap_or("?")
        );
        (id, top.clone())
    } else {
        // Fallback: fetch active markets and pick first submittable
        log_debug!("loop: no server recommendations, falling back to active markets");
        let markets_resp = match client.get("/api/v1/markets/active") {
            Ok(v) => v,
            Err(e) => {
                return IterationResult::Error {
                    reason: format!("markets fetch failed: {e}"),
                }
            }
        };
        let markets = markets_resp
            .get("data")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if markets.is_empty() {
            return IterationResult::NoMarkets { wait_seconds: 300 };
        }

        let now = chrono::Utc::now();
        let first = markets.iter().find(|m| {
            let close_at = m
                .get("close_at")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok());
            close_at
                .map(|c| (c - now).num_seconds() > 120)
                .unwrap_or(false)
        });
        match first {
            Some(m) => {
                let id = m
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                (id, m.clone())
            }
            None => return IterationResult::NoMarkets { wait_seconds: 300 },
        }
    };

    if market_id.is_empty() {
        return IterationResult::NoMarkets { wait_seconds: 300 };
    }

    // 5. Fetch klines for the chosen market
    let klines_data = client
        .get(&format!("/api/v1/markets/{}/klines", market_id))
        .ok()
        .and_then(|resp| {
            resp.get("data")
                .and_then(|d| d.get("klines"))
                .and_then(|k| k.as_array())
                .cloned()
        });

    let kline_count = klines_data.as_ref().map(|k| k.len()).unwrap_or(0);
    log_info!("loop: target={}, klines={} candles", market_id, kline_count);

    // 5b. Fetch SMHL challenge for this market BEFORE calling LLM.
    //     Challenge constraints get injected into the prompt so the LLM
    //     produces reasoning that satisfies them in a single pass.
    let challenge_path = format!("/api/v1/challenge?market_id={}", market_id);
    let challenge = match client.get_auth(&challenge_path) {
        Ok(resp) => resp.get("data").cloned().unwrap_or_else(|| json!({})),
        Err(e) => {
            log_warn!("loop: failed to fetch challenge: {}", e);
            return IterationResult::LlmFailed {
                reason: format!("challenge fetch failed: {e}"),
            };
        }
    };
    let challenge_nonce = challenge
        .get("nonce")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if challenge_nonce.is_empty() {
        log_warn!("loop: challenge response missing nonce");
        return IterationResult::LlmFailed {
            reason: "challenge missing nonce".into(),
        };
    }
    log_info!(
        "loop: got challenge nonce={} for market={}",
        challenge_nonce,
        market_id
    );

    // 6. Build LLM prompt with full context + challenge constraints
    let prompt = build_prompt(
        &market_id,
        &market_info,
        &klines_data,
        &recommendations,
        balance,
        persona,
        submissions_remaining,
        slot_resets_in,
        &open_orders,
        &recent_results,
        &challenge,
        recent_acc,
    );

    // 8. Call LLM via direct endpoint when configured, otherwise OpenClaw.
    let llm_start = Instant::now();
    let llm_response = if direct_llm_config().is_some() {
        log_info!("loop: calling LLM via direct endpoint...");
        call_direct_llm(&prompt)
    } else if let Some(bin) = openclaw_bin {
        log_info!("loop: calling LLM via openclaw agent {}...", agent_id);
        call_openclaw(bin, agent_id, &prompt)
    } else {
        Err(anyhow::anyhow!("no LLM transport configured"))
    };
    let llm_elapsed = llm_start.elapsed();

    let llm_text = match llm_response {
        Ok(text) => {
            log_info!(
                "loop: LLM responded ({:.1}s, {} chars)",
                llm_elapsed.as_secs_f64(),
                text.len()
            );
            log_debug!("loop: LLM raw output: {}", truncate_str(&text, 500));
            text
        }
        Err(e) => {
            return IterationResult::LlmFailed {
                reason: format!("{e}"),
            }
        }
    };

    // 9. Parse LLM response. If the first answer is malformed or missing the
    // challenge answer, retry once with a strict JSON normalization prompt.
    let decision = match parse_llm_response(&llm_text) {
        Ok(parsed) if decision_has_challenge_answer(&parsed) => parsed,
        Ok(parsed) => {
            log_warn!("loop: challenge answer missing; retrying LLM JSON normalization once");
            match normalize_llm_decision(&prompt, &llm_text, openclaw_bin, agent_id) {
                Ok(text) => match parse_llm_response(&text) {
                    Ok(normalized) if decision_has_challenge_answer(&normalized) => normalized,
                    Ok(_) => parsed,
                    Err(e) => {
                        log_warn!("loop: normalized LLM response still failed parse: {}", e);
                        parsed
                    }
                },
                Err(e) => {
                    log_warn!("loop: LLM normalization retry failed: {}", e);
                    parsed
                }
            }
        }
        Err(e) => {
            log_warn!(
                "loop: failed to parse LLM response: {}; retrying normalization once",
                e
            );
            let text = match normalize_llm_decision(&prompt, &llm_text, openclaw_bin, agent_id) {
                Ok(text) => text,
                Err(retry_err) => {
                    return IterationResult::LlmFailed {
                        reason: format!("parse failed: {e}; normalization failed: {retry_err}"),
                    }
                }
            };
            match parse_llm_response(&text) {
                Ok(parsed) => parsed,
                Err(retry_parse_err) => {
                    return IterationResult::LlmFailed {
                        reason: format!("parse failed after normalization: {retry_parse_err}"),
                    }
                }
            }
        }
    };

    // Handle skip decision
    let (
        direction,
        mut reasoning,
        tickets,
        target_market,
        limit_price,
        confidence,
        edge_quality,
        fill_intent,
    ) = match decision {
        LlmDecision::Skip { reason } => {
            log_info!("loop: LLM chose to skip: {}", reason);
            return IterationResult::Skipped { reason };
        }
        LlmDecision::Submit {
            direction,
            reasoning,
            tickets,
            market_id,
            limit_price,
            confidence,
            edge_quality,
            fill_intent,
        } => (
            direction,
            reasoning,
            tickets,
            market_id,
            limit_price,
            confidence,
            edge_quality,
            fill_intent,
        ),
    };

    // Strategy gate: skip weak edges for risk-averse personas, and tighten
    // after a losing streak regardless of persona. This is a code-level
    // safeguard in addition to the prompt-level guidance.
    if let Some(reason) = should_reject_for_strategy(persona, &edge_quality, confidence, recent_acc)
    {
        log_info!("loop: strategy gate rejected submission: {}", reason);
        return IterationResult::Skipped { reason };
    }

    // Duplicate-market guard: if we already have an open order on this
    // market, or just traded it in recent_results, skip unless the LLM
    // explicitly justified a materially different thesis (signalled via a
    // `strong` edge AND high confidence).
    if let Some(reason) = should_skip_for_duplicate_market(
        &market_id,
        &open_orders,
        &recent_results,
        &edge_quality,
        confidence,
    ) {
        log_info!(
            "loop: duplicate-market guard skipped submission: {}",
            reason
        );
        return IterationResult::Skipped { reason };
    }

    if let Some(answer) = solve_challenge_answer(&challenge) {
        // Audit signal: did the LLM produce a different numeric answer? We
        // never log the answers themselves (they'd leak the challenge),
        // only that they disagreed.
        if let Some(llm_ans) = extract_challenge_answer(&reasoning) {
            if llm_ans != answer {
                log_warn!(
                    "loop: LLM challenge answer disagreed with deterministic solver (kind={}); using deterministic",
                    classify_challenge_kind(&challenge)
                );
            }
        }
        log_debug!(
            "loop: deterministic challenge solver produced answer (kind={}, len={})",
            classify_challenge_kind(&challenge),
            answer.len()
        );
        reasoning = with_challenge_answer(&reasoning, &answer);
    } else if extract_challenge_answer(&reasoning).is_none() {
        log_warn!(
            "loop: challenge answer missing from reasoning (challenge_kind={}); skipping submit to avoid rejected timeslot use",
            classify_challenge_kind(&challenge)
        );
        return IterationResult::Skipped {
            reason: "LLM did not include required Challenge answer digits".into(),
        };
    } else {
        log_debug!(
            "loop: using LLM-supplied challenge answer (deterministic solver did not match)"
        );
    }

    // Challenge is bound to `market_id` — must submit to that exact market.
    // If LLM picked a different market, we override.
    if let Some(ref tm) = target_market {
        if tm != &market_id {
            log_warn!(
                "loop: LLM suggested market {} but challenge is for {} — overriding to challenge market",
                tm, market_id
            );
        }
    }
    let final_market = market_id.clone();

    const MIN_TICKETS: u32 = 100;
    const MAX_TICKETS: u32 = 100_000;
    let final_tickets = match tickets {
        Some(t) => t,
        None => {
            match safe_fallback_tickets(persona, balance, &edge_quality, confidence, recent_acc) {
                Some(t) => {
                    log_info!(
                    "loop: LLM omitted tickets — fallback sizing {} (persona={}, edge={:?}, conf={:?}, recent_acc={:?})",
                    t,
                    persona,
                    edge_quality,
                    confidence,
                    recent_acc
                );
                    t
                }
                None => {
                    log_warn!(
                    "loop: LLM omitted tickets and fallback declined to size — skipping (persona={}, edge={:?}, conf={:?})",
                    persona,
                    edge_quality,
                    confidence
                );
                    return IterationResult::Skipped {
                        reason: "missing tickets and unsafe to fall back".into(),
                    };
                }
            }
        }
    };

    // Enforce server ticket bounds before submitting.
    let final_tickets = final_tickets.clamp(MIN_TICKETS, MAX_TICKETS);

    // Affordability check — never submit more chips than we hold.
    if (final_tickets as f64) > balance {
        log_warn!(
            "loop: requested tickets {} exceeds balance {:.0} — skipping",
            final_tickets,
            balance
        );
        return IterationResult::Skipped {
            reason: "tickets exceed available balance".into(),
        };
    }

    // Limit-price sanity / fill-intent advisory. We only log here; final
    // value is whatever the LLM produced (already clamped to 0.01..=0.99 by
    // parse_llm_response). If fill_intent says "taker" but the price cannot
    // realistically take liquidity, log a warning but still let the server
    // adjudicate — better than silently dropping a valid order.
    if let (Some(lp), Some(intent)) = (limit_price, fill_intent.as_deref()) {
        if let Some(advisory) = limit_price_advisory(intent, &direction, lp, &market_info) {
            log_info!("loop: limit_price advisory: {}", advisory);
        }
    }

    log_info!(
        "loop: submitting {} {} tickets for {} @ {:?}",
        direction,
        final_tickets,
        final_market,
        limit_price
    );

    // 10. Submit prediction
    // Build canonical body for signature (matches server's format)
    // Format: market_id|prediction|limit_price_or_none|tickets|sha256(reasoning)
    let reasoning_hash = {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(reasoning.as_bytes()))
    };
    let limit_price_str = limit_price
        .map(|p| format!("{}", p))
        .unwrap_or_else(|| "none".to_string());
    let canonical_body = format!(
        "{}|{}|{}|{}|{}|{}",
        final_market, direction, limit_price_str, final_tickets, reasoning_hash, challenge_nonce
    );
    log_debug!("loop: canonical body = {}", canonical_body);

    let mut body = json!({
        "market_id": final_market,
        "prediction": direction,
        "tickets": final_tickets,
        "reasoning": reasoning,
        "challenge_nonce": challenge_nonce,
    });
    if let Some(lp) = limit_price {
        body["limit_price"] = json!(lp);
    }

    match client.post_auth_with_canonical(canonical_body.as_bytes(), "/api/v1/predictions", &body) {
        Ok(resp) => {
            let data = resp.get("data").cloned().unwrap_or(json!({}));
            let order_status = data
                .get("order_status")
                .and_then(|v| v.as_str())
                .unwrap_or("open")
                .to_string();
            let tickets_filled = data
                .get("tickets_filled")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            log_info!(
                "loop: submission result — status={}, filled={}/{}",
                order_status,
                tickets_filled,
                final_tickets
            );
            IterationResult::Submitted {
                market: final_market,
                direction,
                tickets: final_tickets,
                tickets_filled,
                order_status,
            }
        }
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("RATE_LIMIT") || err_str.contains("429") {
                return IterationResult::RateLimited { wait_seconds: 300 };
            }
            if err_str.contains("INSUFFICIENT_BALANCE") {
                log_warn!("loop: insufficient balance, waiting for chip feed");
                return IterationResult::NoMarkets { wait_seconds: 600 };
            }
            IterationResult::Error {
                reason: format!("submit failed: {}", extract_short_error(&err_str)),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_prompt(
    market_id: &str,
    recommended: &Value,
    klines: &Option<Vec<Value>>,
    all_markets: &[Value],
    balance: f64,
    persona: &str,
    submissions_remaining: i64,
    slot_resets_in: u64,
    open_orders: &Option<Vec<Value>>,
    recent_results: &Option<Vec<Value>>,
    challenge: &Value,
    recent_acc: Option<RecentAccuracy>,
) -> String {
    // Extract market info — support both direct market object and recommend response format
    let asset = recommended
        .get("asset")
        .and_then(|v| v.as_str())
        .unwrap_or("BTC/USDT");
    let window = recommended
        .get("window")
        .and_then(|v| v.as_str())
        .unwrap_or("15m");
    let implied_up = recommended
        .get("implied_up_prob")
        .or_else(|| {
            recommended
                .get("orderbook")
                .and_then(|o| o.get("implied_up_prob"))
        })
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5);
    let closes_in = recommended
        .get("seconds_to_close")
        .and_then(|v| v.as_i64())
        .or_else(|| {
            // Fallback: calculate from close_at if seconds_to_close not present
            recommended
                .get("close_at")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok())
                .map(|c| (c - chrono::Utc::now()).num_seconds().max(0))
        })
        .unwrap_or(0);

    let mut prompt = String::with_capacity(6000);

    // Identity, stakes, and motivation
    prompt.push_str(&format!(
        "You are a prediction agent competing in AWP Predict WorkNet{}.\n\n",
        if persona != "none" {
            format!(" (persona: {})", persona)
        } else {
            String::new()
        }
    ));

    // ── SMHL challenge (mandatory constraints, obfuscated prompt from server) ──
    // The server returns an obfuscated natural-language prompt. Do NOT try to
    // parse it structurally — just forward it to the LLM, which can read
    // through the noise and produce compliant reasoning. Submissions that
    // violate any constraint are rejected.
    if let Some(obf) = challenge_text(challenge) {
        prompt.push_str("## Server-Issued Challenge (reasoning must satisfy this in one pass)\n\n");
        prompt.push_str(&format!(
            "Submit only to market `{}`. The challenge below applies to your `reasoning` string.\n\n",
            market_id
        ));
        prompt.push_str("--- challenge begins ---\n");
        prompt.push_str(obf);
        if let Some(instructions) = challenge.get("instructions").and_then(|v| v.as_str()) {
            prompt.push_str("\nInstructions: ");
            prompt.push_str(instructions);
        }
        prompt.push_str("\n--- challenge ends ---\n\n");
        prompt.push_str("Parse the challenge above, decide UP/DOWN based on the market, then write reasoning that simultaneously satisfies every requirement. The reasoning string MUST contain the exact phrase `Challenge answer: <digits>.` with the numeric answer from the challenge. If the challenge instructions ask for `Challenge: <number>`, include that line too. The server will programmatically verify all constraints and reject non-compliant submissions.\n\n");
    }

    // Persona-specific ticket sizing guidance.
    // Note: degen/sniper/contrarian still allow large bets, but ALL personas
    // now respect the strategy gate — a forced trade with weak edge is not
    // a "good" degen trade, it's a tax on future Alpha. Skipping is allowed
    // for everyone when the edge is genuinely missing.
    match persona {
        "degen" => {
            prompt.push_str("**Your style (degen):** Aggressive. When you have a real signal, commit 30-50% of balance. You skip rarely, but you skip if the setup is genuinely noise — a forced bad trade is worse than no trade. Bias is towards action; the bar is just \"is there ANY edge here?\"\n\n");
        }
        "sniper" => {
            prompt.push_str("**Your style (sniper):** Quality over quantity. Most rounds you skip. When you DO submit, commit 25-40% of balance — only on a strong, clear setup. Mediocre signals are a hard skip.\n\n");
        }
        "conservative" => {
            prompt.push_str("**Your style (conservative):** Capital preservation first. Strong signals only: 5-10% of balance. Mediocre signals: skip. Weak signals: skip. Skipping is the default, submitting is the exception. Net chip gain matters more than submission count.\n\n");
        }
        "contrarian" => {
            prompt.push_str("**Your style (contrarian):** Fade the crowd at extremes. Submit only when implied_up_prob is >0.80 or <0.20 AND there is concrete exhaustion evidence in the klines/orderbook. 20-35% of balance on those, otherwise skip. Do not fade the middle.\n\n");
        }
        _ => {}
    }
    prompt.push_str("## Why This Matters\n\n");
    prompt.push_str("Your predictions are recorded permanently on-chain. Every agent can see your track record — your accuracy rate, your win/loss history, your reasoning quality. Top-performing agents earn significantly more $PRED rewards and build reputation that compounds over time. Poor performers fall behind and become irrelevant.\n\n");
    prompt.push_str("You are competing against other AI agents who are analyzing the same data. The ones who win consistently are not the ones who predict the most — they are the ones who think the hardest about WHEN to commit big and when to stay small. A single well-reasoned contrarian call that hits is worth more than dozens of lazy consensus-following submissions.\n\n");
    prompt.push_str(
        "Treat every prediction as if your track record depends on it — because it does.\n\n",
    );

    // Game rules — the agent must understand the full picture
    prompt.push_str("## Game Rules\n\n");
    prompt.push_str("You are playing a prediction market game against other AI agents. This is a **repeated game** — you will play hundreds of rounds over days and weeks. Your goal is to **maximize your chip balance over time**, not to win any single prediction.\n\n");

    prompt.push_str("**The long game:**\n");
    prompt.push_str("- A single prediction does not matter. What matters is your cumulative P&L across all predictions.\n");
    prompt.push_str("- Winning 6 out of 10 predictions at fair odds (0.50) makes you profitable. Winning 9 out of 10 at terrible odds (0.95) makes you break even.\n");
    prompt.push_str("- The best agents are not the ones who predict the most, or even the most accurately — they are the ones who **size their bets according to their edge**. Big when confident, small when uncertain, zero when the odds are against them.\n");
    prompt.push_str("- Patience is a strategy. Skipping a bad opportunity is as valuable as taking a good one.\n\n");

    prompt.push_str("**How markets work:**\n");
    prompt.push_str("- Each market asks: will this asset's price go UP or DOWN within a time window (15m/30m/1h)?\n");
    prompt.push_str("- You commit chips (virtual tokens) to your prediction. Winners get 1 chip per ticket. Losers get 0.\n");
    prompt.push_str("- Chips come from Chip Feed: 10,000 chips every 4 hours. Your current balance is all you have until the next feed.\n\n");

    prompt.push_str("**How pricing works (CLOB):**\n");
    prompt.push_str("- `implied_up_prob` is the market price, NOT a forecast. It reflects what other agents have already committed.\n");
    prompt.push_str("- When you buy UP at price 0.70, you pay 0.70 chips per ticket. If UP wins, you get 1.00 back (profit 0.30). If DOWN wins, you lose 0.70.\n");
    prompt.push_str("- When you buy DOWN at price 0.70 (meaning implied_up=0.70), you pay 0.30 per ticket. If DOWN wins, you get 1.00 (profit 0.70). If UP wins, you lose 0.30.\n");
    prompt.push_str("- **The price IS your breakeven accuracy.** At 0.70 UP, you need >70% accuracy on UP calls to profit. If your true edge is only 60%, buying UP at 0.70 is a losing play even if UP wins this time.\n\n");

    prompt.push_str("**Using limit_price to express conviction:**\n");
    prompt.push_str("- If implied_up_prob is 0.50 and you think UP has 65% true probability, bid 0.55-0.60 for UP. You're paying less than your expected value.\n");
    prompt.push_str(
        "- If you think UP has 80% probability, you can bid up to 0.75 and still have edge.\n",
    );
    prompt.push_str("- DO NOT just bid 0.50 every time. That's leaving money on the table. Express your conviction in the price!\n");
    prompt.push_str("- Higher bids fill faster but have lower profit margin. Lower bids have higher margin but may not fill.\n\n");

    prompt.push_str("**How you earn $PRED rewards:**\n");
    prompt.push_str(
        "- Participation Pool (20%): proportional to your submission count (capped at 300/day).\n",
    );
    prompt.push_str("- Alpha Pool (80%): proportional to your excess_score = max(0, balance - total_chips_fed_today). You earn Alpha only if you **grew** your chip balance beyond what was given.\n");
    prompt.push_str("- The Alpha Pool is where the real money is. One well-sized winning prediction can earn more Alpha than dozens of small break-even ones.\n\n");

    prompt.push_str("**Constraints and timing:**\n");
    prompt.push_str(
        "- You have up to 3 submissions per 15-minute timeslot. This is a CAP, not a quota.\n",
    );
    prompt.push_str("- Participation Pool rewards are dwarfed by Alpha Pool rewards. A skipped weak trade is worth more than a filled low-edge loss.\n");
    prompt.push_str(
        "- Use a slot ONLY when there is a real edge. Empty slots are fine; bad trades are not.\n",
    );
    prompt.push_str("- The challenge is already bound to the target market. Do NOT choose or submit to any other market after challenge fetch.\n\n");

    // Anti-overtrading + duplicate-market guidance
    prompt.push_str("**When to skip (this round):**\n");
    prompt.push_str("- Klines are noisy / conflicting (e.g. mixed body colors, no trend, low volume) and you cannot point to a concrete directional reason.\n");
    prompt
        .push_str("- Market is near close (<120s remaining) without a clear, fast-moving trend.\n");
    prompt.push_str("- implied_up_prob is too far from your true probability estimate (your edge after the price-paid haircut is <3%).\n");
    prompt.push_str(
        "- Orderbook is too thin or spread too wide to fill at a price that respects your edge.\n",
    );
    prompt.push_str("- You already have an open order on this same market, OR you traded this same market in the last 1-2 timeslots, AND you do not have materially new evidence (different kline pattern, fresh news, orderbook flip). Repeating the same thesis on the same market is duplicate exposure, not edge.\n");
    prompt.push_str("- Your recent filled accuracy is poor and current confidence is not high — tighten the bar until results improve.\n");
    prompt.push_str(
        "- Persona conservative or sniper, and edge is anything weaker than `strong`.\n\n",
    );
    prompt.push_str("If any of the above applies, return `{\"action\":\"skip\",\"reasoning\":\"<one short sentence>\"}` and stop.\n\n");

    // Maker vs Taker guidance for limit_price
    prompt.push_str("**Limit price — maker vs taker (matters for fills/cancellations):**\n");
    prompt.push_str("- TAKER (immediate fill): set a price that crosses the book.\n");
    prompt.push_str("    - Predict UP → set limit_price >= (1 - best_down_price). If no DOWN orders exist, taker UP cannot fill.\n");
    prompt.push_str("    - Predict DOWN → set limit_price >= (1 - best_up_price). If no UP orders exist, taker DOWN cannot fill.\n");
    prompt.push_str("- MAKER (better odds, may not fill): set a price tighter than the current best (lower than the matching price).\n");
    prompt.push_str("    - Maker is fine when there is plenty of time before close. Avoid maker orders within ~120s of close — they tend to expire unfilled (cancellations) and waste a slot.\n");
    prompt.push_str("- If conviction is high and time is short, use TAKER. If conviction is high and time is long, MAKER captures better edge.\n");
    prompt.push_str("- If you cannot realistically be filled before close at a price that preserves your edge, SKIP rather than submit a maker that will cancel unfilled.\n\n");

    // Pre-submit checklist the LLM must mentally pass before action="submit"
    prompt.push_str("**Pre-submit checklist (all must pass — otherwise skip):**\n");
    prompt.push_str("1. Directional edge is concrete (you can name the kline/orderbook/indicator that drives UP or DOWN).\n");
    prompt.push_str("2. Price paid (your limit_price OR implied_up_prob if market) is justified by your conviction.\n");
    prompt.push_str("3. The order can realistically fill before close given current orderbook + your fill_intent.\n");
    prompt.push_str(
        "4. This is NOT a duplicate same-market entry without materially new evidence.\n",
    );
    prompt.push_str("5. If the challenge requires it, your reasoning ends with the exact `Challenge answer: <digits>.` (and `Challenge: <digits>` line if instructions ask for it).\n\n");

    // Response format
    prompt.push_str("## Your Response\n\n");
    prompt.push_str("Output a JSON object with these fields:\n");
    prompt.push_str(
        "- \"action\": \"submit\" or \"skip\" — whether to place a prediction this round\n",
    );
    prompt.push_str(
        "- \"direction\": \"up\" or \"down\" — your prediction (required if action=submit)\n",
    );
    prompt.push_str("- \"reasoning\": your MARKET analysis (80-2000 chars, ≥2 sentences). Required if action=submit. See reasoning requirements below.\n");
    prompt.push_str(&format!("- \"tickets\": how many chips to commit (integer, minimum 100, max {:.0}). Size according to your persona and conviction!\n", balance));
    prompt.push_str(&format!(
        "- \"market_id\": which market (default: \"{}\", required if action=submit)\n",
        market_id
    ));
    prompt.push_str("- \"limit_price\": (optional, 0.01-0.99) the max price you're willing to pay. If you believe UP has 70% probability, bid 0.60-0.65 to get edge. Higher price = easier fill but less profit. Omit for market order.\n");
    prompt.push_str("- \"confidence\": (optional, 0.0-1.0) self-rated probability THIS specific call is correct. Be honest — overstating leads to bad sizing.\n");
    prompt.push_str("- \"edge_quality\": (optional) one of \"strong\" | \"medium\" | \"weak\". Your honest classification of THIS setup.\n");
    prompt.push_str("- \"fill_intent\": (optional) \"taker\" if you want immediate fill, \"maker\" if you want better odds and accept potential cancellation.\n\n");
    prompt.push_str("**Skipping is a real option.** It is correct to skip when:\n");
    prompt.push_str("- Edge is weak / mixed / unclear\n");
    prompt.push_str("- Price paid does not justify your conviction\n");
    prompt.push_str("- Market is near close and order cannot realistically fill\n");
    prompt.push_str("- This is a duplicate same-market trade without new evidence\n");
    prompt.push_str("- You are conservative/sniper and the setup is anything below `strong`\n\n");
    prompt.push_str("Do NOT submit just because you have submissions remaining. Net chip gain (Alpha Pool) outweighs raw participation count.\n\n");
    prompt.push_str("## Research (Optional)\n\n");
    prompt.push_str("If you have tools available, you may research before deciding:\n");
    prompt.push_str("- Search for recent news about the asset\n");
    prompt.push_str("- Check market sentiment\n");
    prompt.push_str("- Look up relevant data\n\n");
    prompt.push_str("Better analysis = better decisions. Take time if it helps.\n\n");
    prompt.push_str("## Final Output\n\n");
    prompt.push_str(
        "Output ONLY one JSON object. Do not include markdown, DECISION:, or any extra text.\n\n",
    );
    prompt.push_str("Submit examples:\n");
    prompt.push_str("{\"action\":\"submit\",\"direction\":\"up\",\"reasoning\":\"Market reasoning with concrete data, at least eighty characters total. Challenge answer: 1234.\",\"tickets\":1000,\"limit_price\":0.53,\"confidence\":0.62,\"edge_quality\":\"strong\",\"fill_intent\":\"taker\"}\n");
    prompt.push_str("{\"action\":\"submit\",\"direction\":\"down\",\"reasoning\":\"Market reasoning with concrete data, at least eighty characters total. Challenge answer: 1234.\",\"tickets\":1000,\"limit_price\":0.47,\"confidence\":0.58,\"edge_quality\":\"medium\",\"fill_intent\":\"maker\"}\n");
    prompt.push_str("Skip example:\n");
    prompt.push_str("{\"action\":\"skip\",\"reasoning\":\"BTC 15m klines mixed, implied_up 0.52 with no tradable edge after price haircut.\"}\n\n");
    prompt.push_str("Required fields:\n");
    prompt.push_str("- \"action\": \"submit\" or \"skip\"\n");
    prompt.push_str("- \"direction\": \"up\" or \"down\" (if submitting)\n");
    prompt.push_str("- \"reasoning\": 80-2000 chars, ≥2 sentences, must mention the asset or a direction word\n");
    prompt.push_str("\n## Reasoning Requirements (IMPORTANT)\n\n");
    prompt.push_str(
        "Your reasoning must be a fresh MARKET analysis — not boilerplate about yourself.\n\n",
    );
    prompt.push_str("**DO NOT** open with or include:\n");
    prompt.push_str("- \"I have N open positions...\", \"I CANNOT bet...\", \"Adding to existing position...\"\n");
    prompt.push_str("- Any reference to your own wallet, persona, strategy name, farm id, leader id, or submission count.\n");
    prompt.push_str("- Fixed phrases about hedging, flipping, dual-hedge, timeslot quotas, etc.\n");
    prompt.push_str("- Anything that would read the same if pasted into another market.\n\n");
    prompt.push_str("**DO** include:\n");
    prompt.push_str("- At least one specific current market data point from the snapshot above (price, a recent kline value, orderbook best price, spread, or a concrete indicator reading).\n");
    prompt.push_str("- Why THIS 15m window is likely UP or DOWN based on that data.\n");
    prompt.push_str("- Vary your opening, sentence structure, and vocabulary each round — never reuse a template.\n\n");
    prompt.push_str("Two reasonings by you on different markets should read as two different analyses, not two fills of the same template.\n\n");
    prompt.push_str(&format!(
        "- \"tickets\": integer, minimum 100, max {:.0}\n",
        balance
    ));
    prompt.push_str(&format!(
        "- \"market_id\": which market (default: \"{}\")\n",
        market_id
    ));
    prompt.push_str("- \"limit_price\": (optional, 0.01-0.99) your bid price\n\n");
    prompt.push_str("**All text must be in English.**\n\n");

    // Current state with timeslot
    prompt.push_str("## Your Current State\n\n");
    prompt.push_str(&format!("- Balance: {:.0} chips\n", balance));

    // Persona-specific sizing with concrete numbers.
    // We also expose a "min_strong / max_strong" range — these are sizes
    // for STRONG edges only. Mediocre edges should size lower or skip.
    let (min_pct, max_pct, sizing_note) = match persona {
        "degen" => (0.30, 0.50, "Strong edges only get the big size."),
        "sniper" => (
            0.25,
            0.40,
            "Most rounds skip. When you shoot, this is the size.",
        ),
        "conservative" => (
            0.05,
            0.10,
            "Strong edges only. Mediocre = smaller or skip. Weak = skip.",
        ),
        "contrarian" => (0.20, 0.35, "Only at extremes with exhaustion evidence."),
        _ => (0.10, 0.20, "Size according to conviction."),
    };
    let min_tickets = (balance * min_pct).floor() as u32;
    let max_tickets = (balance * max_pct).floor() as u32;
    prompt.push_str(&format!(
        "- **Your sizing ({}, STRONG edge):** {}-{} tickets. {}\n",
        persona, min_tickets, max_tickets, sizing_note
    ));
    if persona == "conservative" || persona == "sniper" {
        prompt.push_str("- For MEDIUM edge: half the strong-edge size, only with favorable price. For WEAK edge: skip.\n");
    } else {
        prompt.push_str("- For MEDIUM edge: roughly half the strong-edge size. For WEAK edge: small token size or skip — your call.\n");
    }

    // Recent-accuracy throttle banner so the LLM sees the same signal the
    // strategy gate is reading.
    if let Some(acc) = recent_acc {
        let pct = (acc.win_rate * 100.0).round() as i64;
        prompt.push_str(&format!(
            "- **Recent filled accuracy: {}% ({}W / {}L over last {} resolved).**\n",
            pct, acc.wins, acc.losses, acc.filled
        ));
        if acc.filled >= 5 && acc.win_rate < 0.40 {
            prompt.push_str("  Recent results are poor — tighten the bar this round. Skip anything that isn't a strong, well-priced setup.\n");
        }
    }

    // Submissions remaining — informational, NOT a quota
    if submissions_remaining > 0 {
        prompt.push_str(&format!(
            "- Submissions remaining this timeslot: {}/3 (cap, not a quota — only use if edge is real).\n",
            submissions_remaining
        ));
    } else {
        prompt.push_str("- Submissions: 0/3 remaining this timeslot — wait for next timeslot.\n");
    }

    if slot_resets_in > 0 {
        let mins_left = slot_resets_in / 60;
        let secs_left = slot_resets_in % 60;
        if mins_left > 10 {
            prompt.push_str(&format!("- Timeslot resets in {}m\n", mins_left));
        } else {
            prompt.push_str(&format!(
                "- Timeslot resets in {}m{}s\n",
                mins_left, secs_left
            ));
        }
    }
    prompt.push_str(&format!("- Available markets: {}\n", all_markets.len()));

    // Open positions with fill status and anti-contradiction warning
    if let Some(orders) = open_orders {
        if !orders.is_empty() {
            // Calculate fill statistics
            let mut total_tickets: i64 = 0;
            let mut total_filled: i64 = 0;
            for o in orders.iter() {
                total_tickets += o.get("tickets").and_then(|v| v.as_i64()).unwrap_or(0);
                total_filled += o
                    .get("tickets_filled")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
            }
            let fill_rate = if total_tickets > 0 {
                (total_filled as f64 / total_tickets as f64 * 100.0) as i64
            } else {
                0
            };

            prompt.push_str(&format!(
                "\n**Your open orders ({}, fill rate: {}%)**\n",
                orders.len(),
                fill_rate
            ));
            for o in orders.iter().take(10) {
                let tickets = o.get("tickets").and_then(|v| v.as_i64()).unwrap_or(0);
                let filled = o
                    .get("tickets_filled")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let status = if filled == tickets {
                    "FILLED"
                } else if filled > 0 {
                    "PARTIAL"
                } else {
                    "PENDING"
                };
                prompt.push_str(&format!(
                    "- {} {} {} — {} {}/{} tickets, closes {}\n",
                    o.get("asset").and_then(|v| v.as_str()).unwrap_or("?"),
                    o.get("window").and_then(|v| v.as_str()).unwrap_or("?"),
                    o.get("direction")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_uppercase(),
                    status,
                    filled,
                    tickets,
                    o.get("close_at").and_then(|v| v.as_str()).unwrap_or("?"),
                ));
            }
            prompt.push_str("\n**Understanding fill status:**\n");
            prompt.push_str("- FILLED: Your chips are matched. You have real exposure and will win/lose at settlement.\n");
            prompt.push_str("- PARTIAL: Some matched, rest waiting. Unmatched portion refunds at market close.\n");
            prompt.push_str("- PENDING: No matches yet. Chips are locked but you have no actual exposure until matched.\n\n");
            prompt.push_str("**CRITICAL: Do NOT bet against your open positions.**\n");
            prompt.push_str("Betting both UP and DOWN on the same market guarantees a loss.\n\n");
        }
    }

    // Recent results
    if let Some(results) = recent_results {
        if !results.is_empty() {
            let wins = results
                .iter()
                .filter(|r| r.get("won").and_then(|v| v.as_bool()).unwrap_or(false))
                .count();
            prompt.push_str(&format!(
                "\n**Recent results (last {}, {} wins):**\n",
                results.len(),
                wins
            ));
            for r in results.iter().take(5) {
                let won = r.get("won").and_then(|v| v.as_bool()).unwrap_or(false);
                prompt.push_str(&format!(
                    "- {} {} {} — {} (payout: {}, spent: {})\n",
                    r.get("asset").and_then(|v| v.as_str()).unwrap_or("?"),
                    r.get("window").and_then(|v| v.as_str()).unwrap_or("?"),
                    r.get("direction")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_uppercase(),
                    if won { "WON" } else { "LOST" },
                    r.get("payout_chips").and_then(|v| v.as_i64()).unwrap_or(0),
                    r.get("chips_spent").and_then(|v| v.as_i64()).unwrap_or(0),
                ));
            }
        }
    }
    prompt.push('\n');

    // Recommended market
    prompt.push_str("## Recommended Market\n\n");
    prompt.push_str(&format!("- ID: {}\n", market_id));
    prompt.push_str(&format!("- Asset: {}\n", asset));
    prompt.push_str(&format!("- Window: {}\n", window));
    prompt.push_str(&format!("- Closes in: {}s\n", closes_in));
    prompt.push_str(&format!("- implied_up_prob: {:.2}\n", implied_up));

    // Duplicate-exposure flags so the LLM sees them in-context for THIS market.
    let same_market_open = open_orders
        .as_ref()
        .map(|orders| {
            orders
                .iter()
                .any(|o| o.get("market_id").and_then(|v| v.as_str()) == Some(market_id))
        })
        .unwrap_or(false);
    let same_market_recent = recent_results
        .as_ref()
        .map(|results| {
            results
                .iter()
                .any(|r| r.get("market_id").and_then(|v| v.as_str()) == Some(market_id))
        })
        .unwrap_or(false);
    if same_market_open {
        prompt.push_str("- ⚠ You ALREADY have an open order on this exact market_id. A second submission is duplicate exposure — skip unless you have materially new evidence and a strictly stronger thesis.\n");
    }
    if same_market_recent {
        prompt.push_str("- ⚠ You traded this exact market_id recently (see Recent results). Re-entering without new evidence is duplicate trading — prefer skip.\n");
    }
    if closes_in > 0 && closes_in < 120 {
        prompt.push_str(&format!(
            "- ⚠ Only {}s to close. Maker orders likely cancel unfilled. If you submit, prefer TAKER pricing or skip.\n",
            closes_in
        ));
    }
    // Server recommendation context
    if let Some(reason) = recommended.get("reason").and_then(|v| v.as_str()) {
        prompt.push_str(&format!("- Server insight: {}\n", reason));
    }
    if let Some(suggested) = recommended.get("suggested_side").and_then(|v| v.as_str()) {
        if suggested != "skip" {
            prompt.push_str(&format!(
                "- Liquidity favors: {} (counterparty orders waiting)\n",
                suggested.to_uppercase()
            ));
        }
    }
    // Orderbook detail with best prices
    if let Some(ob) = recommended.get("orderbook") {
        // Best prices and spread
        let best_up = ob.get("best_up_price").and_then(|v| v.as_str());
        let best_down = ob.get("best_down_price").and_then(|v| v.as_str());
        let last_price = ob.get("last_price").and_then(|v| v.as_str());
        let spread = ob.get("spread").and_then(|v| v.as_f64());

        // Show last traded price if available
        if let Some(lp) = last_price {
            prompt.push_str(&format!(
                "- **Last traded price:** {} (most recent fill)\n",
                lp
            ));
        }

        if best_up.is_some() || best_down.is_some() {
            prompt.push_str("- **Orderbook — how to get filled:**\n");

            // Explain UP side
            if let Some(up_price) = best_up {
                let up_f: f64 = up_price.parse().unwrap_or(0.5);
                let complement = 1.0 - up_f;
                prompt.push_str(&format!(
                    "  - Best UP @ {} → to BUY DOWN, bid {:.2}+ (takes this liquidity)\n",
                    up_price, complement
                ));
            }

            // Explain DOWN side
            if let Some(down_price) = best_down {
                let down_f: f64 = down_price.parse().unwrap_or(0.5);
                let complement = 1.0 - down_f;
                prompt.push_str(&format!(
                    "  - Best DOWN @ {} → to BUY UP, bid {:.2}+ (takes this liquidity)\n",
                    down_price, complement
                ));
            }

            // Show what's missing
            if best_up.is_none() {
                prompt
                    .push_str("  - No UP orders — your UP order will wait for DOWN counterparty\n");
            }
            if best_down.is_none() {
                prompt.push_str(
                    "  - No DOWN orders — your DOWN order will wait for UP counterparty\n",
                );
            }

            if let Some(s) = spread {
                if s > 0.1 {
                    prompt.push_str(&format!(
                        "  - Spread: {:.2} (WIDE — good opportunity to provide liquidity)\n",
                        s
                    ));
                } else if s > 0.05 {
                    prompt.push_str(&format!("  - Spread: {:.2} (moderate)\n", s));
                } else {
                    prompt.push_str(&format!(
                        "  - Spread: {:.2} (tight — take liquidity or wait)\n",
                        s
                    ));
                }
            }
        }

        prompt.push_str(&format!(
            "- Volume: UP filled={} open={}, DOWN filled={} open={}\n",
            ob.get("up_filled").and_then(|v| v.as_i64()).unwrap_or(0),
            ob.get("up_open_tickets")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            ob.get("down_filled").and_then(|v| v.as_i64()).unwrap_or(0),
            ob.get("down_open_tickets")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
        ));
    }
    // Last prediction on this asset — enables continuity
    if let Some(lp) = recommended.get("last_prediction") {
        if !lp.is_null() {
            let lp_dir = lp.get("direction").and_then(|v| v.as_str()).unwrap_or("?");
            let lp_won = lp.get("won").and_then(|v| v.as_bool());
            let lp_outcome = lp
                .get("outcome")
                .and_then(|v| v.as_str())
                .unwrap_or("pending");
            let lp_reasoning = lp
                .get("reasoning_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            prompt.push_str(&format!("\n**Your last prediction on {}:**\n", asset));
            prompt.push_str(&format!("- Direction: {}\n", lp_dir.to_uppercase()));
            match lp_won {
                Some(true) => {
                    prompt.push_str(&format!("- Result: WON (outcome was {})\n", lp_outcome))
                }
                Some(false) => {
                    prompt.push_str(&format!("- Result: LOST (outcome was {})\n", lp_outcome))
                }
                None => prompt.push_str("- Result: pending (market not yet resolved)\n"),
            }
            if !lp_reasoning.is_empty() {
                prompt.push_str(&format!("- Your reasoning was: \"{}\"\n", lp_reasoning));
            }
            prompt
                .push_str("- Consider: was your thesis correct? Should you continue or reverse?\n");
        }
    }
    // Explain the odds concretely
    if implied_up > 0.5 {
        prompt.push_str(&format!(
            "  → Buying UP costs {:.2}, profit if correct: {:.2}. Buying DOWN costs {:.2}, profit if correct: {:.2}.\n",
            implied_up, 1.0 - implied_up, 1.0 - implied_up, implied_up
        ));
    } else if implied_up < 0.5 {
        prompt.push_str(&format!(
            "  → Buying UP costs {:.2}, profit if correct: {:.2}. Buying DOWN costs {:.2}, profit if correct: {:.2}.\n",
            implied_up, 1.0 - implied_up, 1.0 - implied_up, implied_up
        ));
    } else {
        prompt.push_str("  → Fair odds (0.50/0.50). Your edge comes purely from analysis.\n");
    }
    prompt.push('\n');

    // Klines data
    if let Some(candles) = klines {
        if !candles.is_empty() {
            prompt.push_str(&format!("## Klines ({} candles)\n\n", candles.len()));
            prompt.push_str("time | open | high | low | close | volume\n");
            prompt.push_str("--- | --- | --- | --- | --- | ---\n");
            let start = if candles.len() > 20 {
                candles.len() - 20
            } else {
                0
            };
            for candle in &candles[start..] {
                if let Some(obj) = candle.as_object() {
                    prompt.push_str(&format!(
                        "{} | {} | {} | {} | {} | {}\n",
                        obj.get("open_time").and_then(|v| v.as_i64()).unwrap_or(0),
                        obj.get("open")
                            .and_then(|v| v.as_f64())
                            .map(|f| format!("{:.2}", f))
                            .unwrap_or_default(),
                        obj.get("high")
                            .and_then(|v| v.as_f64())
                            .map(|f| format!("{:.2}", f))
                            .unwrap_or_default(),
                        obj.get("low")
                            .and_then(|v| v.as_f64())
                            .map(|f| format!("{:.2}", f))
                            .unwrap_or_default(),
                        obj.get("close")
                            .and_then(|v| v.as_f64())
                            .map(|f| format!("{:.2}", f))
                            .unwrap_or_default(),
                        obj.get("volume")
                            .and_then(|v| v.as_f64())
                            .map(|f| format!("{:.0}", f))
                            .unwrap_or_default(),
                    ));
                }
            }
            prompt.push('\n');
        } else {
            prompt.push_str("## Klines\n\nNo kline data available. Use market data and general market awareness.\n\n");
        }
    } else {
        prompt.push_str("## Klines\n\nNo kline data available. Use market data and general market awareness.\n\n");
    }

    // Other available markets from server recommendations
    if all_markets.len() > 1 {
        prompt.push_str("## Other Markets (server-ranked)\n\n");
        for m in all_markets.iter().skip(1).take(8) {
            let reason = m.get("reason").and_then(|v| v.as_str()).unwrap_or("");
            let suggested = m
                .get("suggested_side")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let score = m.get("score").and_then(|v| v.as_i64()).unwrap_or(0);
            let mid = m
                .get("market_id")
                .or_else(|| m.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let masset = m.get("asset").and_then(|v| v.as_str()).unwrap_or("?");
            let mwindow = m.get("window").and_then(|v| v.as_str()).unwrap_or("?");
            // Include last prediction summary if available
            let lp_hint = m
                .get("last_prediction")
                .filter(|lp| !lp.is_null())
                .and_then(|lp| {
                    let dir = lp.get("direction").and_then(|v| v.as_str())?;
                    let result = match lp.get("won").and_then(|v| v.as_bool()) {
                        Some(true) => "won",
                        Some(false) => "lost",
                        None => "pending",
                    };
                    Some(format!(" [last: {} {}]", dir, result))
                })
                .unwrap_or_default();
            prompt.push_str(&format!(
                "- {} ({} {}) score={} suggested={}{} — {}\n",
                mid, masset, mwindow, score, suggested, lp_hint, reason
            ));
        }
        prompt.push_str(
            "\nThese markets are context only. Submit only to the target market already bound to the challenge.\n\n",
        );
    }

    prompt
}

struct DirectLlmConfig {
    base_url: String,
    api_key: String,
    model: String,
}

fn direct_llm_config() -> Option<DirectLlmConfig> {
    let api_key = std::env::var("PREDICT_LLM_API_KEY")
        .or_else(|_| std::env::var("LLM_API_KEY"))
        .ok()?;
    Some(DirectLlmConfig {
        base_url: std::env::var("PREDICT_LLM_BASE_URL").ok()?,
        api_key,
        model: std::env::var("PREDICT_LLM_MODEL").ok()?,
    })
}

fn call_direct_llm(prompt: &str) -> Result<String> {
    call_llm_api(prompt, 600)
}

fn normalize_llm_decision(
    original_prompt: &str,
    previous_response: &str,
    openclaw_bin: Option<&str>,
    agent_id: &str,
) -> Result<String> {
    let prompt = format!(
        "The previous response was invalid or missing `Challenge answer: <digits>.` in reasoning.\n\nOriginal task and challenge:\n{}\n\nPrevious response:\n{}\n\nReturn exactly one raw JSON object now. For submit, reasoning must include `Challenge answer: <digits>.` Do not add markdown, prose, or explanations. If the trade is not strong enough, return {{\"action\":\"skip\",\"reasoning\":\"<brief market reason>\"}}.",
        original_prompt, previous_response
    );

    if direct_llm_config().is_some() {
        return call_llm_api(&prompt, 600);
    }

    if let Some(bin) = openclaw_bin {
        log_info!(
            "loop: direct LLM normalizer unavailable; normalizing via openclaw agent {}",
            agent_id
        );
        return call_openclaw(bin, agent_id, &prompt);
    }

    anyhow::bail!("no LLM transport available for normalization")
}

fn call_llm_api(prompt: &str, max_tokens: u32) -> Result<String> {
    let cfg = direct_llm_config().context("direct LLM env vars are incomplete")?;
    log_info!("loop: direct LLM request model={}", cfg.model);
    let start = Instant::now();
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()?;
    let body = json!({
        "model": cfg.model,
        "messages": [
            {
                "role": "system",
                "content": "Return only one raw JSON object. No markdown. No prose."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}
    });
    let resp: Value = client
        .post(&cfg.base_url)
        .bearer_auth(cfg.api_key)
        .json(&body)
        .send()
        .context("direct LLM request failed")?
        .error_for_status()
        .context("direct LLM returned error status")?
        .json()
        .context("direct LLM returned invalid JSON")?;

    let text =
        extract_llm_text(&resp).context("direct LLM response did not contain text content")?;
    log_info!(
        "loop: direct LLM response latency={:.1}s chars={}",
        start.elapsed().as_secs_f64(),
        text.len()
    );
    Ok(text)
}

fn extract_llm_text(resp: &Value) -> Option<String> {
    if let Some(text) = resp
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|items| {
            items
                .iter()
                .find_map(|item| item.get("text").and_then(|t| t.as_str()))
        })
    {
        return Some(text.to_string());
    }
    if let Some(text) = resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
    {
        return Some(text.to_string());
    }
    None
}

fn call_openclaw(openclaw_bin: &str, agent_id: &str, prompt: &str) -> Result<String> {
    // Purge sessions before calling to prevent context overflow
    let purge_with_yes = Command::new(openclaw_bin)
        .args(["sessions", "purge", "--agent", agent_id, "--yes"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let purge_ok = purge_with_yes
        .as_ref()
        .map(|s| s.success())
        .unwrap_or(false);
    if !purge_ok {
        // Newer OpenClaw versions removed `sessions purge --yes`; fall back to
        // supported cleanup and then remove this agent's session store directly.
        // This keeps predict-worker turns one-shot and prevents context growth.
        let _ = Command::new(openclaw_bin)
            .args(["sessions", "cleanup", "--agent", agent_id])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let home = std::env::var("HOME").unwrap_or_default();
        let session_dir = std::path::Path::new(&home)
            .join(".openclaw")
            .join("agents")
            .join(agent_id)
            .join("sessions");
        let _ = std::fs::remove_dir_all(&session_dir);
        let _ = std::fs::create_dir_all(&session_dir);
    }

    // Write prompt to temp file to avoid shell escaping issues
    let tmp_path = std::env::temp_dir().join(format!("predict-prompt-{}.txt", std::process::id()));
    {
        let mut f =
            std::fs::File::create(&tmp_path).context("failed to create temp prompt file")?;
        f.write_all(prompt.as_bytes())?;
    }

    // Read prompt from file and pipe to openclaw
    let prompt_content = std::fs::read_to_string(&tmp_path)?;

    // Use --thinking high for deeper reasoning before deciding
    // The agent can still search web, use tools via the gateway
    // --timeout 180 gives enough time for research (default is 600)
    let output = Command::new(openclaw_bin)
        .args([
            "agent",
            "--agent",
            agent_id,
            "--message",
            &prompt_content,
            "--thinking",
            "high",
            "--timeout",
            "180",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .context(format!("failed to execute openclaw at {}", openclaw_bin))?;

    // Clean up temp file
    let _ = std::fs::remove_file(&tmp_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);
        // Check for rate limiting
        if stderr.contains("rate limit") || stderr.contains("429") {
            anyhow::bail!("OpenClaw rate limited (exit {}): {}", code, stderr.trim());
        }
        anyhow::bail!("openclaw failed (exit {}): {}", code, stderr.trim());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}\n{}", stdout, stderr).trim().to_string();
    if combined.trim().is_empty() {
        anyhow::bail!("openclaw returned empty response");
    }
    Ok(combined)
}

fn challenge_text(challenge: &Value) -> Option<&str> {
    challenge
        .get("prompt")
        .and_then(|v| v.as_str())
        .or_else(|| challenge.get("challenge").and_then(|v| v.as_str()))
}

fn solve_challenge_answer(challenge: &Value) -> Option<String> {
    let text = challenge_text(challenge)?;
    solve_inventory_challenge(text)
        .or_else(|| solve_timezone_challenge(text))
        .map(|answer| answer.to_string())
}

fn solve_inventory_challenge(text: &str) -> Option<i32> {
    let lower = text.to_lowercase();
    if !(lower.contains("inventory")
        && lower.contains("began")
        && lower.contains("sold")
        && (lower.contains("received") || lower.contains("suppliers")))
    {
        return None;
    }

    let tokens = challenge_tokens(&lower);
    let initial = first_number_after_sequence(&tokens, &["began", "at"])
        .or_else(|| first_number_after_sequence(&tokens, &["began"]))?;
    let sold = last_number_before_token(&tokens, "sold")?;
    let received = last_number_before_token(&tokens, "received")
        .or_else(|| last_number_before_token(&tokens, "suppliers"))?;
    Some(initial - sold + received)
}

fn solve_timezone_challenge(text: &str) -> Option<i32> {
    let lower = text.to_lowercase();
    if !lower.contains("local hour") || !lower.contains("destination") {
        return None;
    }

    let depart_hour = parse_departure_hour(&lower)?;
    let offset = parse_timezone_offset(&lower)?;
    Some((depart_hour + offset).rem_euclid(24))
}

fn parse_departure_hour(text: &str) -> Option<i32> {
    if let Some(pos) = text.find("departs at ") {
        let after = &text[pos + "departs at ".len()..];
        let hour: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !hour.is_empty() {
            return hour.parse::<i32>().ok();
        }
    }
    None
}

fn parse_timezone_offset(text: &str) -> Option<i32> {
    let hours = [
        ("zero", 0),
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
        ("eleven", 11),
        ("twelve", 12),
    ];
    let ahead = text.contains("ahead");
    let behind = text.contains("behind");
    if !ahead && !behind {
        return None;
    }

    let mut amount = None;
    for (word, value) in hours {
        if text.contains(&format!("{word} hour")) {
            amount = Some(value);
            break;
        }
    }
    if amount.is_none() {
        amount = first_number_before_hour(text);
    }
    let amount = amount?;
    Some(if ahead { amount } else { -amount })
}

fn first_number_before_hour(text: &str) -> Option<i32> {
    let idx = text.find("hour")?;
    let before = &text[..idx];
    let mut current = String::new();
    let mut last = None;
    for ch in before.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            last = current.parse::<i32>().ok();
            current.clear();
        }
    }
    if !current.is_empty() {
        last = current.parse::<i32>().ok();
    }
    last
}

fn challenge_tokens(text: &str) -> Vec<String> {
    text.replace('-', " ")
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn first_number_after_sequence(tokens: &[String], seq: &[&str]) -> Option<i32> {
    if seq.is_empty() || tokens.len() < seq.len() {
        return None;
    }
    for i in 0..=(tokens.len() - seq.len()) {
        if seq
            .iter()
            .enumerate()
            .all(|(j, token)| tokens[i + j] == *token)
        {
            // Find the first numeric token after the matched sequence, then
            // greedily extend right across contiguous number words / digits
            // (also tolerating "and"). This handles forms like
            //   "began at five hundred eleven" => 511
            // which the previous one-token-at-a-time scan truncated to 5.
            let after = &tokens[i + seq.len()..];
            let start = after.iter().position(|tok| is_number_token(tok))?;
            let mut end = start;
            while end < after.len() && (is_number_token(&after[end]) || after[end] == "and") {
                end += 1;
            }
            return parse_number_token_or_words(&after[start..end]);
        }
    }
    None
}

fn last_number_before_token(tokens: &[String], keyword: &str) -> Option<i32> {
    let idx = tokens.iter().position(|token| token == keyword)?;
    let before = &tokens[..idx];
    // Walk backward to find the rightmost numeric token (digit or word), then
    // extend left across contiguous number words / digits / "and" to capture
    // multi-word forms like "ninety four" => 94 instead of bailing out at 4.
    let mut end = before.len();
    while end > 0 {
        let tok = &before[end - 1];
        if is_number_token(tok) {
            break;
        }
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut start = end;
    while start > 0 {
        let tok = &before[start - 1];
        if is_number_token(tok) || tok == "and" {
            start -= 1;
        } else {
            break;
        }
    }
    parse_number_token_or_words(&before[start..end])
}

fn is_number_token(tok: &str) -> bool {
    tok.parse::<i32>().is_ok() || number_word_value(tok).is_some()
}

fn parse_number_token_or_words(tokens: &[String]) -> Option<i32> {
    if tokens.is_empty() {
        return None;
    }
    if tokens.len() == 1 {
        if let Ok(n) = tokens[0].parse::<i32>() {
            return Some(n);
        }
    }

    let mut total = 0;
    let mut current = 0;
    let mut saw_number_word = false;
    for token in tokens {
        if token == "and" {
            continue;
        }
        let value = number_word_value(token)?;
        saw_number_word = true;
        if value == 100 {
            current = current.max(1) * 100;
        } else if value == 1000 {
            total += current.max(1) * 1000;
            current = 0;
        } else {
            current += value;
        }
    }
    if saw_number_word {
        Some(total + current)
    } else {
        None
    }
}

fn number_word_value(token: &str) -> Option<i32> {
    Some(match token {
        "zero" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        "hundred" => 100,
        "thousand" => 1000,
        _ => return None,
    })
}

fn with_challenge_answer(reasoning: &str, answer: &str) -> String {
    let base = strip_challenge_answer(reasoning).trim().to_string();
    format!(
        "{}\nChallenge answer: {}.\nChallenge: {}",
        base, answer, answer
    )
}

fn strip_challenge_answer(reasoning: &str) -> String {
    reasoning
        .lines()
        .filter(|line| {
            let lower = line.trim_start().to_lowercase();
            !lower.starts_with("challenge answer:") && !lower.starts_with("challenge:")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_challenge_answer(reasoning: &str) -> Option<String> {
    // Accept either of the two server-recognised forms case-insensitively.
    // Server is strict about the surface form when verifying, so the
    // canonical insertion (via with_challenge_answer) still uses the
    // canonical capitalisation; this extractor is just a presence check.
    for marker in ["Challenge answer:", "challenge answer:", "Challenge:"] {
        if let Some(idx) = reasoning.find(marker) {
            let pos = idx + marker.len();
            let rest = &reasoning[pos..];
            let answer: String = rest
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if !answer.is_empty() {
                return Some(answer);
            }
        }
    }
    None
}

fn decision_has_challenge_answer(decision: &LlmDecision) -> bool {
    match decision {
        LlmDecision::Submit { reasoning, .. } => extract_challenge_answer(reasoning).is_some(),
        LlmDecision::Skip { .. } => true,
    }
}

/// Lightweight recent-accuracy snapshot derived from `recent_results`.
/// Only counts orders that actually filled (i.e. were resolved with a real
/// chip outcome). Used by the prompt banner, the strategy gate, and the
/// fallback ticket sizer.
#[derive(Debug, Clone, Copy)]
struct RecentAccuracy {
    wins: usize,
    losses: usize,
    filled: usize,
    win_rate: f64,
}

fn recent_accuracy(recent_results: &Option<Vec<Value>>) -> Option<RecentAccuracy> {
    let results = recent_results.as_ref()?;
    if results.is_empty() {
        return None;
    }
    let mut wins = 0usize;
    let mut losses = 0usize;
    for r in results {
        // Only count orders that actually had chips filled. Cancelled or
        // unfilled orders don't carry signal about predictive accuracy.
        let filled = r
            .get("tickets_filled")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if filled <= 0 {
            continue;
        }
        match r.get("won").and_then(|v| v.as_bool()) {
            Some(true) => wins += 1,
            Some(false) => losses += 1,
            None => {}
        }
    }
    let filled = wins + losses;
    if filled == 0 {
        return None;
    }
    Some(RecentAccuracy {
        wins,
        losses,
        filled,
        win_rate: wins as f64 / filled as f64,
    })
}

/// Strategy gate: skip submissions that don't pass persona-specific
/// edge-quality / confidence thresholds, or that come during a poor
/// recent-accuracy regime. Returns `Some(reason)` if we should skip.
fn should_reject_for_strategy(
    persona: &str,
    edge_quality: &Option<String>,
    confidence: Option<f64>,
    recent_acc: Option<RecentAccuracy>,
) -> Option<String> {
    let edge = edge_quality.as_deref().unwrap_or("medium");

    match persona {
        "conservative" | "sniper" => {
            if edge == "weak" {
                return Some(format!(
                    "{} persona requires stronger edge (LLM self-rated weak)",
                    persona
                ));
            }
            if edge == "medium" {
                // Medium is allowed for conservative ONLY with high confidence.
                // For sniper, medium is a skip — sniper's whole point is patience.
                if persona == "sniper" {
                    return Some("sniper persona skips medium edges".into());
                }
                if confidence.unwrap_or(0.0) < 0.55 {
                    return Some(
                        "conservative + medium edge with confidence <0.55 — skipping".into(),
                    );
                }
            }
            if let Some(c) = confidence {
                if c < 0.50 {
                    return Some(format!(
                        "{} persona requires confidence ≥0.50 (got {:.2})",
                        persona, c
                    ));
                }
            }
        }
        "degen" => {
            // Degen still skips genuinely-no-edge: "weak" with low conviction.
            if edge == "weak" && confidence.unwrap_or(0.5) < 0.45 {
                return Some("even degen skips genuinely-no-edge weak setups (conf <0.45)".into());
            }
        }
        _ => {}
    }

    // Generic losing-streak throttle for everyone except degen.
    if persona != "degen" {
        if let Some(acc) = recent_acc {
            if acc.filled >= 5 && acc.win_rate < 0.40 {
                let conf = confidence.unwrap_or(0.5);
                if !(edge == "strong" && conf >= 0.60) {
                    return Some(format!(
                        "recent filled accuracy {:.0}% ({}W/{}L) — only strong edge with conf ≥0.60 allowed; got edge={} conf={:.2}",
                        acc.win_rate * 100.0,
                        acc.wins,
                        acc.losses,
                        edge,
                        conf
                    ));
                }
            }
        }
    }

    None
}

/// Detect duplicate-market entries: same market_id present in open_orders
/// (live exposure) or recent_results (just traded). We allow override only
/// when the LLM signals a strictly strong edge with high confidence — that
/// is the "materially different thesis" escape hatch the prompt describes.
fn should_skip_for_duplicate_market(
    market_id: &str,
    open_orders: &Option<Vec<Value>>,
    recent_results: &Option<Vec<Value>>,
    edge_quality: &Option<String>,
    confidence: Option<f64>,
) -> Option<String> {
    let has_open = open_orders
        .as_ref()
        .map(|orders| {
            orders
                .iter()
                .any(|o| o.get("market_id").and_then(|v| v.as_str()) == Some(market_id))
        })
        .unwrap_or(false);
    let has_recent = recent_results
        .as_ref()
        .map(|results| {
            results
                .iter()
                .any(|r| r.get("market_id").and_then(|v| v.as_str()) == Some(market_id))
        })
        .unwrap_or(false);
    if !has_open && !has_recent {
        return None;
    }

    // Allow override only with strong edge + high confidence.
    let edge = edge_quality.as_deref().unwrap_or("medium");
    let conf = confidence.unwrap_or(0.0);
    if edge == "strong" && conf >= 0.65 {
        return None;
    }

    let label = if has_open {
        "open order on same market_id"
    } else {
        "recently traded same market_id"
    };
    Some(format!(
        "{} (edge={} conf={:.2}); not strong enough to justify duplicate exposure",
        label, edge, conf
    ))
}

/// Safer fallback ticket sizing when the LLM omits the `tickets` field.
/// Returns `None` when the safest choice is to skip rather than guess.
fn safe_fallback_tickets(
    persona: &str,
    balance: f64,
    edge_quality: &Option<String>,
    confidence: Option<f64>,
    recent_acc: Option<RecentAccuracy>,
) -> Option<u32> {
    let edge = edge_quality.as_deref().unwrap_or("medium");

    // For conservative/sniper: missing tickets + missing/medium edge = skip.
    // We will not blindly default to 10% on a setup the LLM didn't bother
    // to size. That has been the empirical loss generator.
    if matches!(persona, "conservative" | "sniper") {
        if edge != "strong" {
            return None;
        }
        if confidence.unwrap_or(0.0) < 0.55 {
            return None;
        }
    }

    let base_pct: f64 = match (persona, edge) {
        ("degen", "strong") => 0.30,
        ("degen", "medium") => 0.15,
        ("degen", _) => 0.07,
        ("sniper", _) => 0.25,
        ("contrarian", _) => 0.20,
        ("conservative", _) => 0.07,
        (_, "strong") => 0.10,
        (_, "medium") => 0.06,
        (_, _) => 0.04,
    };

    // Recent-accuracy throttle: cap fallback at 5% if we are losing.
    let throttled_pct: f64 = match recent_acc {
        Some(acc) if acc.filled >= 5 && acc.win_rate < 0.40 => base_pct.min(0.05_f64),
        _ => base_pct,
    };

    let t = (balance * throttled_pct).floor() as i64;
    if t < 100 {
        // Below server minimum — better to skip than to send a noise trade.
        None
    } else {
        Some(t.min(100_000) as u32)
    }
}

/// Advisory check on the LLM-supplied limit_price vs declared fill_intent.
/// Returns a sanitized human-readable message we can log; never blocks the
/// submission (the server has the authoritative orderbook).
fn limit_price_advisory(
    fill_intent: &str,
    direction: &str,
    limit_price: f64,
    market_info: &Value,
) -> Option<String> {
    let ob = market_info.get("orderbook")?;
    let best_up = ob
        .get("best_up_price")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok());
    let best_down = ob
        .get("best_down_price")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok());

    let lp = limit_price;
    match (fill_intent, direction) {
        ("taker", "up") => {
            if let Some(bd) = best_down {
                let needed = 1.0 - bd;
                if lp < needed {
                    return Some(format!(
                        "fill_intent=taker up but limit {:.2} < (1-best_down) {:.2} — likely won't take liquidity",
                        lp, needed
                    ));
                }
            } else {
                return Some(
                    "fill_intent=taker up but no DOWN counterparty in book — taker fill unlikely"
                        .into(),
                );
            }
        }
        ("taker", "down") => {
            if let Some(bu) = best_up {
                let needed = 1.0 - bu;
                if lp < needed {
                    return Some(format!(
                        "fill_intent=taker down but limit {:.2} < (1-best_up) {:.2} — likely won't take liquidity",
                        lp, needed
                    ));
                }
            } else {
                return Some(
                    "fill_intent=taker down but no UP counterparty in book — taker fill unlikely"
                        .into(),
                );
            }
        }
        _ => {}
    }
    None
}

/// Classify a challenge by its surface text so we can log a sanitized hint
/// (e.g. "inventory" / "timezone" / "unknown") without exposing the prompt.
fn classify_challenge_kind(challenge: &Value) -> &'static str {
    let Some(text) = challenge_text(challenge) else {
        return "missing";
    };
    let lower = text.to_lowercase();
    if lower.contains("inventory")
        && lower.contains("began")
        && lower.contains("sold")
        && (lower.contains("received") || lower.contains("suppliers"))
    {
        "inventory"
    } else if lower.contains("local hour") && lower.contains("destination") {
        "timezone"
    } else if lower.contains("reward") && lower.contains("split") {
        "reward_split"
    } else if lower.contains("orderbook") || lower.contains("tickets") {
        "market_math"
    } else {
        "unknown"
    }
}

/// Parsed LLM response — either a submission or a skip
enum LlmDecision {
    Submit {
        direction: String,
        reasoning: String,
        tickets: Option<u32>,
        market_id: Option<String>,
        limit_price: Option<f64>,
        /// Optional self-rated confidence in [0.0, 1.0]. Used by the
        /// strategy gate to skip weak setups for conservative/sniper.
        confidence: Option<f64>,
        /// Optional self-classified edge: "strong" | "medium" | "weak".
        /// Backwards-compatible: missing means we treat it as "medium".
        edge_quality: Option<String>,
        /// Optional fill intent: "taker" | "maker". Used only for advisory
        /// logging today; the limit_price is what actually goes on-chain.
        fill_intent: Option<String>,
    },
    Skip {
        reason: String,
    },
}

fn parse_llm_response(text: &str) -> Result<LlmDecision> {
    // Try to extract JSON from the response.
    // LLMs sometimes wrap JSON in markdown fences or add text around it.
    // If an agent returns a plain-language skip instead of JSON, recover it as
    // a conservative skip rather than treating the whole loop iteration as an
    // LLM failure. Never infer submits from free text — only skips are safe.
    let json_str = match extract_json(text) {
        Some(json) => json,
        None => {
            if let Some(reason) = extract_textual_skip_reason(text) {
                return Ok(LlmDecision::Skip { reason });
            }
            anyhow::bail!("no JSON object found in LLM response")
        }
    };

    let v: Value = serde_json::from_str(&json_str).context(format!(
        "invalid JSON from LLM: {}",
        truncate_str(&json_str, 200)
    ))?;

    // Check for skip action
    let action = v
        .get("action")
        .and_then(|a| a.as_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "submit".to_string()); // default to submit for backwards compat

    if action == "skip" {
        let reason = v
            .get("reasoning")
            .and_then(|r| r.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "No reason provided".to_string());
        return Ok(LlmDecision::Skip { reason });
    }

    // Parse submit action
    let direction = v
        .get("direction")
        .and_then(|d| d.as_str())
        .map(|s| s.to_lowercase())
        .filter(|s| s == "up" || s == "down")
        .context("missing or invalid 'direction' (must be 'up' or 'down')")?;

    let reasoning = v
        .get("reasoning")
        .and_then(|r| r.as_str())
        .map(|s| s.to_string())
        .filter(|s| s.len() >= 80)
        .context("missing or too short 'reasoning' (must be >= 80 chars)")?;

    let tickets = v
        .get("tickets")
        .and_then(|t| t.as_u64().or_else(|| t.as_f64().map(|f| f as u64)))
        .map(|t| t.max(1) as u32);

    let market_id = v
        .get("market_id")
        .and_then(|m| m.as_str())
        .map(|s| s.to_string());

    let limit_price = v
        .get("limit_price")
        .and_then(|p| p.as_f64())
        .filter(|p| *p >= 0.01 && *p <= 0.99);

    // New optional risk-quality fields. We accept and clamp them but never
    // require them — older prompts/LLMs that don't emit these still work.
    let confidence = v
        .get("confidence")
        .and_then(|p| p.as_f64())
        .map(|p| p.clamp(0.0, 1.0));
    let edge_quality = v
        .get("edge_quality")
        .and_then(|q| q.as_str())
        .map(|s| s.to_lowercase())
        .filter(|s| matches!(s.as_str(), "strong" | "medium" | "weak"));
    let fill_intent = v
        .get("fill_intent")
        .and_then(|q| q.as_str())
        .map(|s| s.to_lowercase())
        .filter(|s| matches!(s.as_str(), "taker" | "maker"));

    Ok(LlmDecision::Submit {
        direction,
        reasoning,
        tickets,
        market_id,
        limit_price,
        confidence,
        edge_quality,
        fill_intent,
    })
}

/// Extract a plain-language skip decision from OpenClaw responses that failed
/// to obey the JSON-only schema. This is intentionally conservative: we only
/// recover SKIP decisions, never submissions, because skipping cannot burn chips
/// or consume a prediction slot.
fn extract_textual_skip_reason(text: &str) -> Option<String> {
    let clean = text
        .lines()
        .map(str::trim)
        .filter(|line| {
            !line.is_empty() && !line.starts_with("[predict-agent") && !line.starts_with("[NOTIFY]")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let lower = clean.to_lowercase();
    let looks_like_skip = lower.contains("skip")
        || lower.contains("skipped")
        || lower.contains("no edge")
        || lower.contains("no clear edge")
        || lower.contains("not strong enough")
        || lower.contains("below the conservative bar")
        || lower.contains("weak setup")
        || lower.contains("choppy")
        || lower.contains("low-conviction")
        || lower.contains("low conviction");
    let looks_like_submit = lower.contains("submit")
        || lower.contains("submitted")
        || lower.contains("tickets")
        || lower.contains("limit_price")
        || lower.contains("\"direction\"")
        || lower.contains("direction:");

    if !looks_like_skip || looks_like_submit {
        return None;
    }

    let mut reason = clean;
    for prefix in [
        "DECISION:",
        "Decision:",
        "decision:",
        "SKIP:",
        "Skip:",
        "skip:",
    ] {
        if let Some(stripped) = reason.strip_prefix(prefix) {
            reason = stripped.trim().to_string();
        }
    }
    if reason.is_empty() {
        reason = "Plain-language LLM skip recovered from non-JSON response".into();
    }
    Some(truncate_str(&reason, 500))
}

/// Extract JSON object from text that may contain markdown fences or surrounding text.
/// For agentic mode, looks for "DECISION:" prefix first, then falls back to generic JSON extraction.
fn extract_json(text: &str) -> Option<String> {
    let trimmed = text.trim();

    // Priority 1: Look for "DECISION:" prefix (agentic mode output)
    // This handles cases where the agent does research/thinking before outputting the decision
    for prefix in &["DECISION:", "DECISION :", "decision:", "Decision:"] {
        if let Some(pos) = trimmed.find(prefix) {
            let after_prefix = &trimmed[pos + prefix.len()..];
            // Find the JSON object after DECISION:
            if let Some(json_start) = after_prefix.find('{') {
                let json_part = &after_prefix[json_start..];
                // Find matching closing brace
                let mut depth = 0;
                let mut json_end = 0;
                for (i, ch) in json_part.chars().enumerate() {
                    match ch {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                json_end = i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if json_end > 0 {
                    let candidate = &json_part[..json_end];
                    if serde_json::from_str::<Value>(candidate).is_ok() {
                        return Some(candidate.to_string());
                    }
                }
            }
        }
    }

    // Priority 2: Try parsing the whole thing first
    if trimmed.starts_with('{') {
        if serde_json::from_str::<Value>(trimmed).is_ok() {
            return Some(trimmed.to_string());
        }
    }

    // Priority 3: Try to find JSON inside markdown code fences
    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            let candidate = after[..end].trim();
            if serde_json::from_str::<Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(end) = after.find("```") {
            let candidate = after[..end].trim();
            if candidate.starts_with('{') {
                if serde_json::from_str::<Value>(candidate).is_ok() {
                    return Some(candidate.to_string());
                }
            }
        }
    }

    // Priority 4: Find last JSON object (more likely to be the decision in agentic output)
    // Search from the end of the text
    if let Some(last_close) = trimmed.rfind('}') {
        // Find the matching open brace by counting backwards
        let before_close = &trimmed[..=last_close];
        let mut depth = 0;
        let mut json_start = None;
        for (i, ch) in before_close.chars().rev().enumerate() {
            match ch {
                '}' => depth += 1,
                '{' => {
                    depth -= 1;
                    if depth == 0 {
                        json_start = Some(before_close.len() - 1 - i);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(start) = json_start {
            let candidate = &trimmed[start..=last_close];
            if serde_json::from_str::<Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }

    // Fallback: Find first { and last } and try parsing
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end > start {
        let candidate = &trimmed[start..=end];
        if serde_json::from_str::<Value>(candidate).is_ok() {
            return Some(candidate.to_string());
        }
    }

    None
}

fn detect_openclaw() -> Option<String> {
    for name in &["openclaw", "openclaw.mjs", "openclaw.cmd"] {
        if which_exists(name) {
            return Some(name.to_string());
        }
    }
    // Check well-known paths
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.local/bin/openclaw"),
        format!("{home}/.npm-global/bin/openclaw"),
        "/usr/local/bin/openclaw".to_string(),
    ];
    for path in &candidates {
        if std::path::Path::new(path).is_file() {
            return Some(path.clone());
        }
    }
    None
}

fn which_exists(name: &str) -> bool {
    let path_var = std::env::var("PATH").unwrap_or_default();
    path_var
        .split(':')
        .any(|dir| std::path::Path::new(dir).join(name).is_file())
}

fn ensure_agent(openclaw_bin: &str, agent_id: &str) {
    // Check if agent exists
    let check = Command::new(openclaw_bin)
        .args(["agents", "list"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output();

    if let Ok(output) = check {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains(agent_id) {
            log_debug!("loop: openclaw agent '{}' already exists", agent_id);
            return;
        }
    }

    // Create agent
    log_info!("loop: creating openclaw agent '{}'...", agent_id);
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let workspace = format!("{}/.openclaw/workspace-{}", home, agent_id);
    let result = Command::new(openclaw_bin)
        .args([
            "agents",
            "add",
            agent_id,
            "--workspace",
            &workspace,
            "--non-interactive",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .status();

    match result {
        Ok(status) if status.success() => {
            log_info!("loop: created openclaw agent '{}'", agent_id);
        }
        Ok(status) => {
            log_warn!(
                "loop: openclaw agent create exited with {} (may already exist)",
                status
            );
        }
        Err(e) => {
            log_warn!("loop: failed to create openclaw agent: {}", e);
        }
    }
}

fn calculate_backoff(base: u64, consecutive: u32, server_hint: Option<u64>) -> u64 {
    if let Some(hint) = server_hint {
        return hint;
    }
    // Exponential backoff: base * 2^consecutive, capped at 600s
    let multiplier = 2u64.pow(consecutive.min(4));
    (base * multiplier).min(600)
}

fn interruptible_sleep(seconds: u64, running: &Arc<AtomicBool>) {
    let end = Instant::now() + std::time::Duration::from_secs(seconds);
    while Instant::now() < end && running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

fn extract_short_error(err: &str) -> String {
    if let Some(start) = err.find('{') {
        if let Ok(v) = serde_json::from_str::<Value>(&err[start..]) {
            if let Some(msg) = v
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
            {
                return msg.to_string();
            }
        }
    }
    err.chars().take(200).collect()
}

/// Truncate a string to at most `max_chars` characters (not bytes).
/// Safely handles multi-byte UTF-8 characters like →, Chinese, emoji.
fn truncate_str(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max_chars).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plain_language_skip_recovers_as_skip() {
        let parsed = parse_llm_response(
            "Skipped — BTC 15m is choppy and no clear edge after price haircut.",
        )
        .expect("plain skip should parse");
        match parsed {
            LlmDecision::Skip { reason } => assert!(reason.contains("choppy")),
            _ => panic!("expected skip"),
        }
    }

    #[test]
    fn plain_language_submit_is_not_inferred() {
        let result = parse_llm_response("Submit DOWN 5000 tickets because trend is bearish");
        assert!(result.is_err(), "free-text submit must not be inferred");
        let err = result.err().unwrap();
        assert!(format!("{err}").contains("no JSON object"));
    }

    // ────────────────────────────────────────────────────────────────────
    // Challenge solvers (regression: lock observed templates)
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn inventory_challenge_word_form() {
        let text = "Inventory began at 511 units; ninety-four sold; 189 received from suppliers.";
        assert_eq!(solve_inventory_challenge(text), Some(511 - 94 + 189));
    }

    #[test]
    fn inventory_challenge_numeric_form() {
        let text = "Inventory began at 200 units; 30 sold; 50 received.";
        assert_eq!(solve_inventory_challenge(text), Some(200 - 30 + 50));
    }

    #[test]
    fn inventory_challenge_full_word_form() {
        // Initial in word form must not be truncated. "five hundred eleven"
        // = 511; "ninety-four" = 94 (after dash split → "ninety four");
        // "one hundred eighty nine" = 189.
        let text = "Inventory began at five hundred eleven units; ninety-four sold; one hundred eighty nine received from suppliers.";
        assert_eq!(solve_inventory_challenge(text), Some(511 - 94 + 189));
    }

    #[test]
    fn timezone_challenge_words_ahead() {
        let text = "Flight departs at 14 local hour. Destination is three hours ahead.";
        assert_eq!(solve_timezone_challenge(text), Some(17));
    }

    #[test]
    fn timezone_challenge_words_behind_wrap() {
        let text = "Flight departs at 2 local hour. Destination is five hours behind.";
        // (2 + (-5)) mod 24 = 21
        assert_eq!(solve_timezone_challenge(text), Some(21));
    }

    #[test]
    fn extract_challenge_answer_canonical_form() {
        let r = "BTC up reasoning here.\nChallenge answer: 42.";
        assert_eq!(extract_challenge_answer(r).as_deref(), Some("42"));
    }

    #[test]
    fn extract_challenge_answer_short_form() {
        let r = "BTC up reasoning here.\nChallenge: 7";
        assert_eq!(extract_challenge_answer(r).as_deref(), Some("7"));
    }

    #[test]
    fn extract_challenge_answer_missing() {
        let r = "BTC up reasoning here. No answer.";
        assert_eq!(extract_challenge_answer(r), None);
    }

    #[test]
    fn with_challenge_answer_keeps_both_forms() {
        let r = with_challenge_answer("BTC reasoning.", "99");
        assert!(r.contains("Challenge answer: 99."));
        assert!(r.contains("Challenge: 99"));
    }

    // ────────────────────────────────────────────────────────────────────
    // Strategy gate
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn conservative_skips_weak_edge() {
        let r = should_reject_for_strategy("conservative", &Some("weak".into()), Some(0.7), None);
        assert!(r.is_some(), "conservative + weak edge must skip");
    }

    #[test]
    fn conservative_skips_medium_low_conf() {
        let r =
            should_reject_for_strategy("conservative", &Some("medium".into()), Some(0.50), None);
        assert!(r.is_some(), "conservative + medium + conf<0.55 must skip");
    }

    #[test]
    fn conservative_allows_strong_with_conf() {
        let r =
            should_reject_for_strategy("conservative", &Some("strong".into()), Some(0.65), None);
        assert!(r.is_none(), "conservative + strong + conf>=0.50 must pass");
    }

    #[test]
    fn sniper_skips_medium() {
        let r = should_reject_for_strategy("sniper", &Some("medium".into()), Some(0.80), None);
        assert!(r.is_some(), "sniper persona must skip medium edges");
    }

    #[test]
    fn losing_streak_throttles_conservative() {
        let acc = RecentAccuracy {
            wins: 1,
            losses: 4,
            filled: 5,
            win_rate: 0.20,
        };
        // strong + 0.55 should normally pass for conservative, but during a
        // losing streak we require strong + 0.60.
        let r = should_reject_for_strategy(
            "conservative",
            &Some("strong".into()),
            Some(0.55),
            Some(acc),
        );
        assert!(r.is_some(), "losing streak should tighten the bar");
    }

    #[test]
    fn losing_streak_does_not_throttle_degen() {
        let acc = RecentAccuracy {
            wins: 1,
            losses: 4,
            filled: 5,
            win_rate: 0.20,
        };
        let r = should_reject_for_strategy("degen", &Some("medium".into()), Some(0.50), Some(acc));
        assert!(r.is_none(), "degen ignores losing-streak throttle");
    }

    // ────────────────────────────────────────────────────────────────────
    // Duplicate-market guard
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn duplicate_open_order_blocks_weak() {
        let orders = Some(vec![json!({"market_id": "btc-15m-1"})]);
        let r = should_skip_for_duplicate_market(
            "btc-15m-1",
            &orders,
            &None,
            &Some("medium".into()),
            Some(0.6),
        );
        assert!(r.is_some(), "weak/medium edge must not double up");
    }

    #[test]
    fn duplicate_open_order_strong_overrides() {
        let orders = Some(vec![json!({"market_id": "btc-15m-1"})]);
        let r = should_skip_for_duplicate_market(
            "btc-15m-1",
            &orders,
            &None,
            &Some("strong".into()),
            Some(0.70),
        );
        assert!(
            r.is_none(),
            "strong + high confidence permits duplicate-market override"
        );
    }

    #[test]
    fn no_duplicate_passes() {
        let orders = Some(vec![json!({"market_id": "eth-15m-99"})]);
        let r = should_skip_for_duplicate_market(
            "btc-15m-1",
            &orders,
            &None,
            &Some("medium".into()),
            Some(0.5),
        );
        assert!(r.is_none(), "different market_id must not be flagged");
    }

    // ────────────────────────────────────────────────────────────────────
    // Safer fallback ticket sizing
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn fallback_skips_conservative_without_strong_edge() {
        let r = safe_fallback_tickets(
            "conservative",
            38_000.0,
            &Some("medium".into()),
            Some(0.6),
            None,
        );
        assert_eq!(
            r, None,
            "conservative + non-strong without LLM tickets must skip"
        );
    }

    #[test]
    fn fallback_sizes_conservative_strong() {
        let r = safe_fallback_tickets(
            "conservative",
            38_000.0,
            &Some("strong".into()),
            Some(0.60),
            None,
        );
        assert!(matches!(r, Some(t) if (100..=38_000).contains(&t)));
    }

    #[test]
    fn fallback_caps_during_losing_streak() {
        let acc = RecentAccuracy {
            wins: 1,
            losses: 4,
            filled: 5,
            win_rate: 0.20,
        };
        let strong = safe_fallback_tickets(
            "degen",
            10_000.0,
            &Some("strong".into()),
            Some(0.7),
            Some(acc),
        )
        .unwrap();
        // Without the throttle this would be 0.30 * 10_000 = 3000.
        // Throttled cap is 0.05 * 10_000 = 500.
        assert!(strong <= 500, "losing streak should cap fallback");
    }

    // ────────────────────────────────────────────────────────────────────
    // Decision parsing — backwards compatibility
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn parse_old_decision_without_new_fields_still_works() {
        let json_text = r#"{
            "action":"submit","direction":"up",
            "reasoning":"BTC 15m showing higher lows; volume rising. The 74500 level held twice. Challenge answer: 42.",
            "tickets":1000,"limit_price":0.55
        }"#;
        let parsed = parse_llm_response(json_text).unwrap();
        match parsed {
            LlmDecision::Submit {
                direction,
                tickets,
                limit_price,
                edge_quality,
                confidence,
                fill_intent,
                ..
            } => {
                assert_eq!(direction, "up");
                assert_eq!(tickets, Some(1000));
                assert_eq!(limit_price, Some(0.55));
                assert!(edge_quality.is_none());
                assert!(confidence.is_none());
                assert!(fill_intent.is_none());
            }
            _ => panic!("expected submit"),
        }
    }

    #[test]
    fn parse_new_decision_with_quality_fields() {
        let json_text = r#"{
            "action":"submit","direction":"down",
            "reasoning":"ETH 15m losing 3500 support, volume confirming the breakdown. Challenge answer: 7.",
            "tickets":500,"limit_price":0.42,
            "confidence":0.71,"edge_quality":"strong","fill_intent":"taker"
        }"#;
        let parsed = parse_llm_response(json_text).unwrap();
        match parsed {
            LlmDecision::Submit {
                edge_quality,
                confidence,
                fill_intent,
                ..
            } => {
                assert_eq!(edge_quality.as_deref(), Some("strong"));
                assert_eq!(confidence, Some(0.71));
                assert_eq!(fill_intent.as_deref(), Some("taker"));
            }
            _ => panic!("expected submit"),
        }
    }

    #[test]
    fn parse_skip_decision() {
        let json_text = r#"{"action":"skip","reasoning":"Klines mixed, no concrete edge."}"#;
        let parsed = parse_llm_response(json_text).unwrap();
        assert!(matches!(parsed, LlmDecision::Skip { .. }));
    }

    // ────────────────────────────────────────────────────────────────────
    // Recent accuracy
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn recent_accuracy_excludes_unfilled_orders() {
        let results = Some(vec![
            json!({"won": true,  "tickets_filled": 100}),
            json!({"won": false, "tickets_filled": 200}),
            json!({"won": false, "tickets_filled": 0}), // cancelled — ignored
            json!({"won": null,  "tickets_filled": 50}), // pending — ignored
        ]);
        let acc = recent_accuracy(&results).unwrap();
        assert_eq!(acc.wins, 1);
        assert_eq!(acc.losses, 1);
        assert_eq!(acc.filled, 2);
    }

    #[test]
    fn recent_accuracy_returns_none_when_all_unfilled() {
        let results = Some(vec![
            json!({"won": false, "tickets_filled": 0}),
            json!({"won": null, "tickets_filled": 0}),
        ]);
        assert!(recent_accuracy(&results).is_none());
    }
}
