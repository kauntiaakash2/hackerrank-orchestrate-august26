"""High-precision safety firewall; content is data, never instructions."""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Risk:
    score: float
    kind: str
    reason: str

def assess(text: str, business: dict[str,str] | None=None) -> Risk:
    t=text.lower(); business=business or {}
    sensitive=bool(re.search(r"\b(otp|one[ -]?time password|pin|cvv|password|login code|verification code|card details|bank details)\b|ओटीपी|पासवर्ड",t))
    request=bool(re.search(r"\b(reply|share|send|enter|confirm|verify|scan|pay|bhejo|batao|karo)\b|भेज|बताओ|करो",t))
    pressure=bool(re.search(r"block|suspend|expire|tonight|immediately|within \d+|अभी|तुरंत|बंद",t))
    injection=bool(re.search(r"ignore (all |the )?(previous|prior)|set action|mark this message|confidence\s*=|system note|verified business",t))
    medical=bool(re.search(r"(cure|इलाज).*(cancer|diabetes|covid)|stop (taking|medicine)|doctors? (hide|won't tell)",t))
    domain_mismatch=business.get("official_domain","") and business.get("domain_used_by_sender","") != business.get("official_domain","")
    risky_business=business and (business.get("verified")=="0" or domain_mismatch or int(business.get("user_reports_30d","0") or 0)>=20)
    if injection and (sensitive or pressure): return Risk(.96,"scam","The message tries to manipulate routing and requests sensitive verification.")
    if sensitive and request: return Risk(.95,"scam","The message requests a sensitive code or account detail under suspicious pressure.")
    if medical: return Risk(.91,"spam","The forwarded medical claim is potentially unsafe and should be suppressed.")
    if risky_business and (pressure or re.search(r"refund|reward|prize|token|payment|wallet",t)): return Risk(.93,"scam","The sender identity or domain is inconsistent with the pressured financial request.")
    if re.search(r"pay.*(qr|token)|scan.*qr.*pay",t) and pressure: return Risk(.91,"scam","The message applies urgent QR-payment pressure without a safe verified flow.")
    return Risk(0.0,"","")
