"""OCPP 1.6J / 2.0.1 + ISO 15118 / IEC 61851 charging-telemetry adapter for REAL public EV charging sessions.

Real data used: "Electric vehicle charging dataset with 35,000 charging sessions from 12 residential locations in Norway" (Zenodo 13896176,
CC-BY 4.0): per session = location, user, plug-in time, plug-out time, connection hours, delivered kWh.
The dataset has no within-session power curve, so `reconstruct_profile` rebuilds one that EXACTLY conserves the real delivered energy and
connection window using a standard constant-current / constant-voltage (CC-CV) shape. Every reconstructed stream carries
meta["profile"] = "reconstructed_from_real_session_energy" (this is a model, not a measurement).

Also provided: an adapter for the ACN-Data session schema (connectionTime, disconnectTime, kWhDelivered, ...), whose API needs a personal token,
so ACN records are not bundled; point `load_acn_json` at your own download.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..streams import Stream

V_PH, V_LL = 230.0, 400.0
CLASSES = {"1ph16A": (1, 16.0), "1ph32A": (1, 32.0), "3ph16A": (3, 16.0)}


@dataclass
class Session:
    session_id: str; location: str; plugin: pd.Timestamp; plugout: pd.Timestamp; energy_kwh: float; source: str = "norway-zenodo-13896176"

    @property
    def hours(self): return (self.plugout - self.plugin).total_seconds() / 3600.0


def load_norway(path) -> list[Session]:
    d = pd.read_csv(path, sep=";", decimal=",", parse_dates=["plugin_time", "plugout_time"])
    d = d[(d.energy_session > 0.1) & (d.plugout_time > d.plugin_time)]
    return [Session(r.session_id, r.location, r.plugin_time, r.plugout_time, float(r.energy_session)) for r in d.itertuples()]


def load_acn_json(path_or_obj) -> list[Session]:
    """ACN-Data JSON schema (`_items`: connectionTime, disconnectTime, kWhDelivered, sessionID, siteID, ...)."""
    obj = json.load(open(path_or_obj)) if isinstance(path_or_obj, str) else path_or_obj
    out = []
    for it in obj.get("_items", obj if isinstance(obj, list) else []):
        t0 = pd.to_datetime(it["connectionTime"]); t1 = pd.to_datetime(it["disconnectTime"])
        out.append(Session(str(it.get("sessionID", it.get("_id"))), str(it.get("siteID", "acn")), t0.tz_localize(None) if t0.tzinfo else t0,
                           t1.tz_localize(None) if t1.tzinfo else t1, float(it["kWhDelivered"]), "acn-data"))
    return out


def rated_class(location_sessions: list[Session]) -> str:
    """Charger class inferred from the data: 99th-percentile average power of the location's sessions (3-phase 16 A = 11.04 kW)."""
    p = np.array([s.energy_kwh / max(s.hours, 1e-3) for s in location_sessions])
    q = np.percentile(p, 99)
    return "1ph16A" if q <= 3.9 else ("1ph32A" if q <= 7.6 else "3ph16A")


def rated_power_kw(cls: str) -> float:
    ph, a = CLASSES[cls]
    return (V_PH * ph * a if ph == 1 else math.sqrt(3) * V_LL * a) / 1000.0


def pilot_duty_pct(current_a: float) -> float:
    """IEC 61851-1 / ISO 15118-2 control-pilot PWM: 10..85 % duty -> I = duty*0.6 A (6..51 A); 85..96 % -> (duty-64)*2.5."""
    return current_a / 0.6 if current_a <= 51 else 64 + current_a / 2.5


def cos_phi(load_frac):
    """Modelling constant (not measured): EVSE on-board charger power factor drops at light load (CV taper). Scalar or array."""
    x = np.asarray(load_frac, dtype=float)
    y = np.clip(0.90 + 0.09 * np.minimum(x / 0.5, 1.0), 0.90, 0.99)
    return float(y) if y.ndim == 0 else y


def peak_power_kw(s: Session, cls: str) -> float:
    """Peak power of the reconstructed CC-CV profile (exceeds the class rating only when the real session is too short/energetic for it)."""
    P = rated_power_kw(cls); T = s.hours * 3600.0; E = s.energy_kwh * 3600.0
    P = max(P, 1.02 * E / T); t1 = 0.8 * E / P; tau = 0.2 * E / P
    if t1 + 4 * tau > T:
        k = T / (t1 + 4 * tau); t1 *= k; tau *= k
        P = E / (t1 + tau * (1 - math.exp(-(T - t1) / max(tau, 1e-9))))
    return P


def reconstruct_profile(s: Session, cls: str, dt_s: int = 5, t_start_s: float = 0.0, t_len_s: float | None = None) -> Stream:
    """CC-CV profile conserving the real delivered energy. Time axis = ms since plug-in (so scenarios can window it)."""
    ph, a_max = CLASSES[cls]; P = rated_power_kw(cls)
    T = s.hours * 3600.0
    E = s.energy_kwh * 3600.0                      # kW*s
    P = max(P, 1.02 * E / T)                       # a session that averages above class rating implies a higher class
    t1 = 0.8 * E / P                               # CC phase length
    tau = 0.2 * E / P                              # CV taper time constant (exact integral = P*tau for T>>tau)
    if t1 + 4 * tau > T:                           # short session: compress CC/CV proportionally, keep energy exact
        k = T / (t1 + 4 * tau); P = P / k if False else P; t1 *= k; tau *= k
        P = E / (t1 + tau * (1 - math.exp(-(T - t1) / max(tau, 1e-9))))
    t_len = T if t_len_s is None else t_len_s
    ts = np.arange(t_start_s, min(t_start_s + t_len, T) + 1e-9, dt_s)
    p = np.where(ts <= t1, P, P * np.exp(-(ts - t1) / max(tau, 1e-9)))
    I = p * 1000.0 / ((V_PH * ph) if ph == 1 else math.sqrt(3) * V_LL * 1.0) / cos_phi(p / P)
    e_cum = np.array([E_t for E_t in _cum(ts, P, t1, tau)]) / 3600.0                           # kWh
    return Stream(f"ocpp:{s.session_id}", "ocpp", (ts * 1000).astype(np.int64),
                  {"power_kw": p, "current_a": I, "voltage_v": np.full_like(ts, V_PH if ph == 1 else V_LL), "energy_kwh": e_cum,
                   "cos_phi": cos_phi(p / P), "pilot_limit_a": np.full_like(ts, a_max),
                   "pilot_duty_pct": np.full_like(ts, pilot_duty_pct(a_max))},
                  {"session_id": s.session_id, "location": s.location, "class": cls, "phases": ph, "rated_a": a_max, "rated_kw": P,
                   "energy_kwh_real": s.energy_kwh, "plugin": s.plugin.isoformat(), "plugout": s.plugout.isoformat(),
                   "profile": "reconstructed_from_real_session_energy", "dataset": s.source})


def _cum(ts, P, t1, tau):
    for t in ts:
        yield P * min(t, t1) + (P * tau * (1 - math.exp(-(t - t1) / max(tau, 1e-9))) if t > t1 else 0.0)


# ---------------------------------------------------------------- OCPP-J frames
def _iso(plugin: pd.Timestamp, t_ms: float) -> str:
    return (plugin + pd.Timedelta(milliseconds=float(t_ms))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def ocpp16_frames(st: Stream, tx_id: int = 1, connector: int = 1, every_s: int = 60) -> list[dict]:
    """OCPP 1.6J CALL frames [2, uniqueId, action, payload] for the session (StartTransaction, MeterValues, StopTransaction)."""
    plugin = pd.Timestamp(st.meta["plugin"]); f = []
    f.append({"t_ms": int(st.t_ms[0]), "frame": [2, f"{tx_id}-start", "StartTransaction",
              {"connectorId": connector, "idTag": "ANON", "meterStart": 0, "timestamp": _iso(plugin, st.t_ms[0])}]})
    step = max(1, int(every_s * 1000 / max(np.median(np.diff(st.t_ms)), 1)))
    for i in range(0, len(st), step):
        c = st.cols
        sv = [("Energy.Active.Import.Register", c["energy_kwh"][i] * 1000, "Wh"), ("Power.Active.Import", c["power_kw"][i] * 1000, "W"),
              ("Current.Import", c["current_a"][i], "A"), ("Voltage", c["voltage_v"][i], "V"), ("Current.Offered", c["pilot_limit_a"][i], "A")]
        f.append({"t_ms": int(st.t_ms[i]), "frame": [2, f"{tx_id}-mv-{i}", "MeterValues", {"connectorId": connector, "transactionId": tx_id,
                  "meterValue": [{"timestamp": _iso(plugin, st.t_ms[i]), "sampledValue": [
                      {"value": f"{v:.2f}", "context": "Sample.Periodic", "measurand": m, "unit": u} for m, v, u in sv]}]}]})
    f.append({"t_ms": int(st.t_ms[-1]), "frame": [2, f"{tx_id}-stop", "StopTransaction",
              {"transactionId": tx_id, "meterStop": int(st.cols["energy_kwh"][-1] * 1000), "timestamp": _iso(plugin, st.t_ms[-1]), "reason": "Local"}]})
    return f


def set_charging_profile(limit_a: float, start_s: float = 0, duration_s: int | None = None, profile_id: int = 1, stack: int = 0) -> list:
    """OCPP 1.6 SetChargingProfile CALL (TxProfile, current limit in A). This is the message the attack scenarios forge."""
    return [2, f"scp-{profile_id}", "SetChargingProfile", {"connectorId": 1, "csChargingProfiles": {
        "chargingProfileId": profile_id, "stackLevel": stack, "chargingProfilePurpose": "TxProfile", "chargingProfileKind": "Relative",
        "chargingSchedule": {"chargingRateUnit": "A", "duration": duration_s, "chargingSchedulePeriod": [{"startPeriod": int(start_s), "limit": limit_a}]}}}]


def to_ocpp201_transaction_event(frame16: list) -> list:
    """Map a 1.6J StartTransaction/MeterValues/StopTransaction frame to its OCPP 2.0.1 TransactionEvent equivalent."""
    _, uid, action, pl = frame16
    ev = {"StartTransaction": "Started", "MeterValues": "Updated", "StopTransaction": "Ended"}[action]
    body = {"eventType": ev, "timestamp": pl.get("timestamp", pl.get("meterValue", [{}])[0].get("timestamp")), "triggerReason":
            {"Started": "CablePluggedIn", "Updated": "MeterValuePeriodic", "Ended": "EVCommunicationLost"}[ev], "seqNo": 0,
            "transactionInfo": {"transactionId": str(pl.get("transactionId", "1"))}}
    if action == "MeterValues":
        body["meterValue"] = [{"timestamp": body["timestamp"], "sampledValue": [{"value": float(x["value"]), "measurand": x["measurand"],
                              "unitOfMeasure": {"unit": x["unit"]}} for x in pl["meterValue"][0]["sampledValue"]]}]
    return [2, uid, "TransactionEvent", body]
