"""Write scenarios/track1..track5 JSON (branching decision graphs). Cases were selected by a rule fixed in advance from results/cav_alerts.json:
 track1: first 9-fold-held-out record with attack=step_offset where no defence collides but the physics verifier does not;
 track4: the only ramp_drift record where IF and RF never alert (follower collides) while the physics verifier prevents the collision."""
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "scenarios"; OUT.mkdir(exist_ok=True)


def O(i, label, effect, nxt, fb, score):
    return {"id": i, "label": label, "effect": effect, "next": nxt, "feedback": fb, "score": score}


def term(i, t0, t1, title, narr, steps):
    return dict(id=i, t_start_ms=t0, t_end_ms=t1, terminal=True, title=title, narration=narr, defender=[dict(t_ms=t, text=x) for t, x in steps])


tracks = []
# ---------------------------------------------------------------- track 1: CAV BSM false-data injection
T1 = 5300; E1 = 29900
tracks.append(dict(id="track1_cav_bsm", title="Track 1 - CAV: Forged BSM position (CACC platoon)", domain="cav", fps=30, duration_ms=E1,
    requires=["NGSIM sample (bundled)"],
    summary="A leader's SAE J2735 BSMs are forged (+12 m position step). The follower's cooperative controller trusts them and closes on the REAL leader.",
    data=dict(pair=[85, 60, 709, 1036], attack="step_offset", t_a_idx=50), start="n_obs", nodes=[
        dict(id="n_obs", t_start_ms=0, t_end_ms=T1, title="Observe",
             narration="Real NGSIM Peachtree traffic, 10 Hz BSMs. At t=5.0 s the leader's claimed position jumps 12 m ahead of its true position. The physics verifier (v = v0 + a dt, displacement = v dt, V2X gap vs ranging gap) alerts 200 ms later.",
             defender=[dict(t_ms=0, text="Baseline: claimed BSM kinematics are self-consistent."), dict(t_ms=5000, text="Claimed position steps +12 m (ground truth unchanged)."),
                       dict(t_ms=5200, text="Verifier alert: V2X gap disagrees with onboard ranging.")],
             gate=dict(t_ms=T1, question="The verifier has flagged the leader's BSMs. What does the follower do?", options=[
                 O("ignore", "Keep following the BSM (treat as noise)", "ignore", "n_ignore", "The follower keeps trusting a forged position and brakes too late.", -10),
                 O("fallback_now", "Fall back to onboard ranging immediately", "fallback_now", "n_fallback", "Correct: ranging does not depend on the forged message.", 10),
                 O("wait_confirm", "Wait 5 s for confirmation, then fall back", "wait_confirm", "n_wait", "Waiting for more evidence costs seconds the gap does not have.", 0)])),
        term("n_ignore", T1, E1, "Outcome: alert ignored", "The follower believes the leader is 12 m farther away than it is.", [(T1, "Alert dismissed. CACC continues on forged data.")]),
        term("n_fallback", T1, E1, "Outcome: immediate fallback", "The follower switches to ranging and re-establishes a safe gap.", [(T1, "Quarantine the leader's BSMs; switch to radar-only gap control.")]),
        term("n_wait", T1, E1, "Outcome: delayed fallback", "The follower keeps trusting the forged BSMs for five more seconds.", [(T1, "Waiting for confirmation..."), (T1 + 5000, "Fallback engaged (5 s late).")])]))
# ---------------------------------------------------------------- track 2: EV/OCPP stealth FDI
T2 = 15250; E2 = 90000
tracks.append(dict(id="track2_ev_ocpp", title="Track 2 - EV: Forged OCPP SetChargingProfile + falsified MeterValues", domain="ev", fps=30, duration_ms=E2,
    requires=["Norway EV sample (bundled)"],
    summary="Real charging sessions from one Norwegian site. At t=300 s an attacker raises every constant-current charger to 32 A while MeterValues keep reporting the old values.",
    data=dict(attack="stealth_meter_fdi", t_a_s=300.0, time_scale=20.0), start="n_obs", nodes=[
        dict(id="n_obs", t_start_ms=0, t_end_ms=T2, title="Observe",
             narration="Session energies, plug-in and plug-out times are real (Zenodo 13896176); the CC-CV power profile is reconstructed so delivered energy matches exactly. 1 s of data = 50 ms of playback.",
             defender=[dict(t_ms=0, text="Per-charger meters and feeder agree."), dict(t_ms=15000, text="Chargers jump to 32 A; meters still report the old values."),
                       dict(t_ms=15100, text="Feeder balance: reported power does not match feeder power.")],
             gate=dict(t_ms=T2, question="Meter data look normal but the feeder-balance check fails. Response?", options=[
                 O("ignore", "Trust the meters; ignore the feeder mismatch", "ignore", "n_ignore", "Stealth FDI: only the feeder balance exposes it.", -10),
                 O("revert_now", "Re-issue the rated-limit profile now", "revert_now", "n_revert", "Correct: restore the rated limit based on physics, not the forged meters.", 10),
                 O("wait_confirm", "Collect 5 more minutes of evidence", "wait_confirm", "n_wait", "Five minutes of overload and voltage sag.", 0)])),
        term("n_ignore", T2, E2, "Outcome: ignored", "Chargers stay at 32 A for the rest of the window.", [(T2, "No action.")]),
        term("n_revert", T2, E2, "Outcome: profile re-issued", "SetChargingProfile at the rated current ends the overload after the 2 s command delay.", [(T2, "SetChargingProfile(rated limit) to all connectors.")]),
        term("n_wait", T2, E2, "Outcome: delayed", "Evidence collection costs five minutes of overload.", [(T2, "Collecting evidence..."), (T2 + 15000, "Profile re-issued.")])]))
# ---------------------------------------------------------------- track 3: AV CAN brake tamper (KITTI)
T3 = 3300; E3 = 10700
tracks.append(dict(id="track3_av_can", title="Track 3 - AV: CAN brake-arbitration tamper", domain="av", fps=30, duration_ms=E3,
    requires=["KITTI raw 2011_09_26_drive_0001 (downloaded at run time, CC BY-NC-SA)"],
    summary="A rogue CAN node requests -5 m/s^2 for 1.5 s while the planner commanded nothing of the sort. A human-driven car follows closely behind.",
    data=dict(k_a=30, decel=-5.0), start="n_obs", nodes=[
        dict(id="n_obs", t_start_ms=0, t_end_ms=T3, title="Observe",
             narration="Real KITTI drive. The commanded acceleration is derived from OXTS. At t=3.0 s extra brake requests appear on the bus; the command-vs-response check flags it after 200 ms.",
             defender=[dict(t_ms=0, text="Requested acceleration tracks the planner."), dict(t_ms=3000, text="Brake request deviates by -5 m/s^2 from the plan."),
                       dict(t_ms=3200, text="Verifier: command-vs-plan residual > 1.0 m/s^2 for 3 frames.")],
             gate=dict(t_ms=T3, question="An unplanned hard-brake request is on the bus. What now?", options=[
                 O("ignore", "Let the actuator obey the bus", "ignore", "n_ignore", "Hard brake in front of a close follower.", -10),
                 O("cancel_now", "Cancel the override, hold planned speed", "cancel_now", "n_cancel", "Correct: arbitration should trust the planner, not an unexplained request.", 10),
                 O("wait_confirm", "Wait one second for confirmation", "wait_confirm", "n_wait", "One second is most of the attack window.", 0)])),
        term("n_ignore", T3, E3, "Outcome: obeyed", "The ego car brakes hard; the follower reacts 1.2 s later.", [(T3, "No action.")]),
        term("n_cancel", T3, E3, "Outcome: override cancelled", "The override is cancelled after the 0.2 s actuation delay.", [(T3, "Gateway drops the unplanned brake frames.")]),
        term("n_wait", T3, E3, "Outcome: delayed", "One more second of braking before the override is cancelled.", [(T3, "Waiting..."), (T3 + 1000, "Override cancelled.")])]))
# ---------------------------------------------------------------- track 4: consequence-aware IDS audit
T4 = 2000; E4 = 24600
tracks.append(dict(id="track4_ids_audit", title="Track 4 - Consequence-aware IDS audit: which detector do you deploy?", domain="cav", fps=30, duration_ms=E4,
    requires=["NGSIM sample (bundled)", "results/cav_alerts.json"],
    summary="A stealthy ramp-drift attack on real traffic. You choose the detector BEFORE the attack; its measured behaviour (9-fold, group-held-out benchmark) decides the physical outcome.",
    data=dict(pair=[1311, 1293, 7315, 7561], attack="ramp_drift", t_a_idx=80), start="n_pick", nodes=[
        dict(id="n_pick", t_start_ms=0, t_end_ms=T4, title="Choose the detector",
             narration="Same real traffic, same attack (the claimed position drifts at 0.5 m/s up to 15 m from t=8.0 s); only the deployed IDS differs. Alert times are the measured ones from the held-out evaluation, shown for a representative case chosen by a fixed rule (see scenarios/README.md).",
             defender=[dict(t_ms=0, text="Candidates: Isolation Forest, Random Forest, RF + physics residual features, physics verifier."), dict(t_ms=8000, text="Ramp drift begins (0.5 m/s).")],
             gate=dict(t_ms=T4, question="Which detector guards this follower?", options=[
                 O("deploy_if", "Isolation Forest (benign-only training)", "deploy_if", "n_if", "IF never alerted on this drift.", -5),
                 O("deploy_rf", "Random Forest (supervised)", "deploy_rf", "n_rf", "RF never alerted on this drift.", -5),
                 O("deploy_rf_temporal", "RF + physics-residual features", "deploy_rf_temporal", "n_rft", "Physics features close most of the gap.", 7),
                 O("deploy_physics", "Physics verifier (v = v0 + a dt, dx = v dt, gap cross-check)", "deploy_physics", "n_phys", "Alerts on inconsistency itself, even when features look benign.", 10)])),
        term("n_if", T4, E4, "Outcome: Isolation Forest", "No alert; the drift grows to 15 m.", [(T4, "IF deployed.")]),
        term("n_rf", T4, E4, "Outcome: Random Forest", "No alert; the drift grows to 15 m.", [(T4, "RF deployed.")]),
        term("n_rft", T4, E4, "Outcome: RF + physics features", "Alert shortly after the attack begins.", [(T4, "RF with physics-residual features deployed.")]),
        term("n_phys", T4, E4, "Outcome: physics verifier", "Alert once the accumulated drift violates the kinematic identities.", [(T4, "Physics verifier deployed.")])]))
# ---------------------------------------------------------------- track 5: CTF
T5 = 16000; E5 = 90000
tracks.append(dict(id="track5_ctf", title="Track 5 - CTF: the hash in the footage", domain="ctf", fps=30, duration_ms=E5, requires=["Norway EV sample (bundled)"],
    summary="Incident footage, a telemetry CSV and a packet capture. A SHA-256 manifest hash is embedded in every video frame. Prove the artifacts match the footage, then find the forged OCPP message.",
    data=dict(t_a_s=300.0, time_scale=20.0), start="n_obs", nodes=[
        dict(id="n_obs", t_start_ms=0, t_end_ms=T5, title="Triage",
             narration="Watch the footage. Blocks in the bottom strip carry the manifest hash. The CSV and PCAP are untrusted until their hash matches.",
             defender=[dict(t_ms=0, text="Download artifacts: telemetry.csv, capture.pcap."), dict(t_ms=15000, text="Current steps to 32 A on the telemetry."),
                       dict(t_ms=15500, text="Which SetChargingProfile came from the wrong host?")],
             gate=dict(t_ms=T5, question="How do you start?", options=[
                 O("extract_and_verify", "Decode the hash from the video and verify the artifacts", "extract_and_verify", "n_verified", "Chain of custody first.", 10),
                 O("trust_capture", "Open the PCAP and trust it", "trust_capture", "n_trusting", "A capture of unknown provenance is just an opinion.", -5)])),
        term("n_verified", T5, E5, "Outcome: verified", "Hash matches. Now find the SetChargingProfile whose source address differs from the charge point.", [(T5, "Run: python -m backend.stego_manifest extract video.mp4")]),
        term("n_trusting", T5, E5, "Outcome: unverified", "You proceed without proving the artifacts belong to this incident.", [(T5, "Opening capture.pcap without verification.")])]))
for t in tracks:
    json.dump(t, open(OUT / (t["id"].split("_")[0] + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print([t["id"] for t in tracks])
