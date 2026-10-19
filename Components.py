"""BOM line semantics: fitted or not, the value written in the Name ("10k", "220uF16V"), component type, and the marking
a resistor of that value carries.

Altium BOM Name conventions on our boards (seen in SocketTester.Main r79):
    NC, DNP                 not fitted
    NC( 7.2k ), nc( 100R )  not fitted; the value in brackets is what would go there
    100R/NC, LD39100 /NC    optional: fitted on some builds, empty on others; the value before /NC
    0/NC, 10k/NC            same
Values: resistors 10k, 1.5k, 620R, 0.2R, 0R, 23.2 (bare = ohms), 4K7 / 2R2 (RKM); capacitors 0.1uF, 10n, 6.8pF,
220uF16V, 10nF50V, 0.1uF25V, 1.2pF (RF); inductors 2.2nH (RF); crystals 12MHz, ABM8-32.000MHZ-9-D1G-T; ferrite beads
120R3A (120 Ω at 100 MHz, 3 A).

classifyLine is the server's classify_line (server.py, "component type of a BOM line") ported unchanged.
"""
import math
import re

# ---------------------------------------------------------------- fitted / not fitted

_NC = re.compile(r"^\s*(?:nc|dnp|dni|n/?f)\s*$", re.I)
_NC_OF = re.compile(r"^\s*(?:nc|dnp)\s*\(\s*(.*?)\s*\)\s*$", re.I)
_OR_NC = re.compile(r"^(.*?)\s*/\s*(?:nc|dnp)\s*$", re.I)


def parseFit(name):
    """BOM Name -> (fit, name of the part that goes there). fit: 'fitted' | 'nc' | 'optional'."""
    name = (name or "").strip()
    if _NC.match(name):
        return "nc", ""
    m = _NC_OF.match(name)
    if m:
        return "nc", m.group(1)
    m = _OR_NC.match(name)
    if m:
        return "optional", m.group(1)
    return "fitted", name


# ---------------------------------------------------------------- values

PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3, "K": 1e3, "M": 1e6,
          "G": 1e9}
_NUM = r"(\d+(?:[.,]\d+)?)"


def _f(s):
    return float(s.replace(",", "."))


def _r(v):
    return float(f"{v:.6g}")


def formatSi(v, unit):
    """1500, 'Ω' -> '1.5 kΩ'; 1e-7, 'F' -> '100 nF'."""
    if v == 0:
        return f"0 {unit}"
    for p, m in (("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("µ", 1e-6), ("n", 1e-9), ("p", 1e-12)):
        if abs(v) >= m * 0.9999:
            return f"{v / m:.4g} {p}{unit}"
    return f"{v:.3g} {unit}"


def parseResistance(s):
    """'10k' 1e4, '4K7' 4700, '2R2' 2.2, '0.2R' 0.2, '23.2' 23.2, '1M' 1e6, '0' 0 -> ohms or None."""
    s = s.strip().replace("Ω", "R").replace("ohm", "R").replace("Ohm", "R")
    m = re.match(r"^(\d+)([RrKkM])(\d+)$", s)  # RKM: 4K7, 2R2
    if m:
        mult = {"R": 1.0, "r": 1.0, "k": 1e3, "K": 1e3, "M": 1e6}[m.group(2)]
        return _r(_f(f"{m.group(1)}.{m.group(3)}") * mult)
    m = re.match(r"^" + _NUM + r"\s*([RrKkMm]?)\b", s)
    if m:
        mult = {"": 1.0, "R": 1.0, "r": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "m": 1e-3}[m.group(2)]
        return _r(_f(m.group(1)) * mult)
    return None


def parseCapacitance(s):
    """'0.1uF' 1e-7, '10n' 1e-8, '4n7' 4.7e-9, '220uF16V' 2.2e-4 -> farads or None."""
    m = re.match(r"^(\d+)([pnuµμ])(\d+)\b", s)  # 4n7
    if m:
        return _r(_f(f"{m.group(1)}.{m.group(3)}") * PREFIX[m.group(2)])
    m = re.match(r"^" + _NUM + r"\s*([pnuµμm]?)F?", s)
    if m and (m.group(2) or "F" in s[:m.end() + 1]):
        return _r(_f(m.group(1)) * PREFIX[m.group(2)])
    return None


def parseValue(name, group, sub=""):
    """BOM Name of a fitted part -> {kind, value (SI), unit, text, ...} or None when the Name is a part number."""
    s = re.sub(r"\((?:rf)\)", "", name or "", flags=re.I).strip()
    if not s:
        return None
    if group == "Resistors" and sub != "thermistors":
        v = parseResistance(s)
        if v is None:
            return None
        out = {"kind": "resistance", "value": v, "unit": "Ω", "text": formatSi(v, "Ω")}
        out["marking"] = resistorMarking(v)
        return out
    if group == "Capacitors":
        v = parseCapacitance(s)
        if v is None:
            return None
        out = {"kind": "capacitance", "value": v, "unit": "F", "text": formatSi(v, "F")}
        # a number directly followed by V, anywhere: 220uF16V, 25V 100nF, 6V3 (RKM: 6.3 V)
        volt = re.search(r"(?<![\d.,])" + _NUM + r"\s*V(\d)?(?![A-Za-z])", s)
        if volt:
            out["voltage"] = _f(volt.group(1) + ("." + volt.group(2) if volt.group(2) else ""))
        die = re.search(r"\b(X[5-8][RSTUV]|Y5V|NP0|C0G|COG)\b", s, re.I)
        if die:
            out["dielectric"] = die.group(1).upper()
        return out
    if group == "Inductors" and sub == "ferrite beads":
        m = re.match(r"^(\d+)\s*R(?:\s*" + _NUM + r"\s*A)?", s)
        if not m:
            return None
        out = {"kind": "impedance", "value": float(m.group(1)), "unit": "Ω", "text": f"{m.group(1)} Ω @ 100 MHz"}
        if m.group(2):
            out["current"] = _f(m.group(2))
        return out
    if group == "Inductors":
        m = re.match(r"^" + _NUM + r"\s*([pnuµμm]?)H\b", s)
        if not m:
            return None
        v = _r(_f(m.group(1)) * PREFIX[m.group(2)])
        return {"kind": "inductance", "value": v, "unit": "H", "text": formatSi(v, "H")}
    if group == "Crystals & oscillators":
        m = re.search(r"(?<![\d.])" + _NUM + r"\s*([kKM]?)HZ", s, re.I)
        if not m:
            return None
        v = _f(m.group(1)) * {"": 1.0, "k": 1e3, "K": 1e3, "M": 1e6}[m.group(2)]
        return {"kind": "frequency", "value": v, "unit": "Hz", "text": formatSi(v, "Hz")}
    return None


# ---------------------------------------------------------------- resistor marking codes

E96 = [100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 137, 140, 143, 147, 150, 154, 158, 162, 165,
       169, 174, 178, 182, 187, 191, 196, 200, 205, 210, 215, 221, 226, 232, 237, 243, 249, 255, 261, 267, 274, 280,
       287, 294, 301, 309, 316, 324, 332, 340, 348, 357, 365, 374, 383, 392, 402, 412, 422, 432, 442, 453, 464, 475,
       487, 499, 511, 523, 536, 549, 562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732, 750, 768, 787, 806,
       825, 845, 866, 887, 909, 931, 953, 976]
EIA96_MULT = {-3: "Z", -2: "Y", -1: "X", 0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}  # 0.01 is also R, 0.1 also S


def _digits(v, n):
    """v -> (n significant digits as int, exponent) when v has at most n significant digits, else None."""
    if v <= 0:
        return None
    e = math.floor(math.log10(v)) - (n - 1)
    mant = v / 10 ** e
    if abs(mant - round(mant)) > 1e-6 * mant:
        return None
    return int(round(mant)), e


def _rCode(v, n):
    """n-character code (3: '103', 4: '1002'); below 10 / 100 Ω the decimal point is an R ('4R7', '23R2', 'R200')."""
    if v >= 10 ** (n - 2):
        d = _digits(v, n - 1)
        return f"{d[0]}{d[1]}" if d else None
    for dec in range(n):
        s = f"{v:.{dec}f}"
        if abs(float(s) - v) < 1e-9 * max(v, 1):
            break
    else:
        return None
    i, _, f = s.partition(".")
    i = i.lstrip("0")
    if len(i) + len(f) > n - 1:
        return None
    return f"{i}R{f}{'0' * (n - 1 - len(i) - len(f))}"


def resistorMarking(ohms):
    """Codes a chip resistor of this value may carry: {'3': '103', '4': '1002', 'eia96': '01C', 'zero': '0'}.
    0402 and smaller are usually unmarked; 0603 1 % parts carry EIA-96; 5 % parts the 3-digit code."""
    out = {}
    if ohms == 0:
        return {"zero": "0", "3": "000"}
    for n in (3, 4):
        c = _rCode(ohms, n)
        if c:
            out[str(n)] = c
    d = _digits(ohms, 3) if ohms >= 10 else None
    if d:
        mant, e = d
        if mant in E96 and e in EIA96_MULT:
            out["eia96"] = f"{E96.index(mant) + 1:02d}{EIA96_MULT[e]}"
    return out


# ---------------------------------------------------------------- component type (server classify_line)

GROUPS = ["Resistors", "Capacitors", "Inductors", "Diodes & protection", "Transistors", "LEDs & optocouplers",
          "ICs", "Crystals & oscillators", "Connectors", "Buttons & switches", "Other"]

# (regex over "Top > Sub > Leaf" of DigiKey, group, subgroup); first match wins; subgroup None -> decided by keywords
DK_RULES = [
    (r"thermistor", "Resistors", "thermistors"),
    (r"^resistors.*array", "Resistors", "arrays"),
    (r"^resistors", "Resistors", "chip resistors"),
    (r"^capacitors > ceramic", "Capacitors", "ceramic"),
    (r"^capacitors > (aluminum|tantalum|polymer)", "Capacitors", "polarized"),
    (r"^capacitors", "Capacitors", "other"),
    (r"^inductors.*ferrite", "Inductors", "ferrite beads"),
    (r"^inductors", "Inductors", "chokes"),
    (r"^circuit protection > tvs|esd", "Diodes & protection", "ESD / TVS"),
    (r"^circuit protection", "Diodes & protection", "other protection"),
    (r"diodes - zener", "Diodes & protection", "Zener"),
    (r"^discrete.*diodes", "Diodes & protection", "rectifier / Schottky"),
    (r"^discrete.*fets", "Transistors", "MOSFET"),
    (r"^discrete.*(bipolar|bjt)", "Transistors", "bipolar"),
    (r"^discrete", "Transistors", "other"),
    (r"^optoelectronics > led", "LEDs & optocouplers", "LEDs"),
    (r"^isolators > opto", "LEDs & optocouplers", "optocouplers"),
    (r"pmic\) > voltage regulators - dc dc", "ICs", "power: DC-DC"),
    (r"pmic\) > voltage regulators - linear|low drop out|ldo", "ICs", "power: LDO"),
    (r"pmic\) > battery", "ICs", "power: battery charger"),
    (r"pmic\)", "ICs", None),
    (r"^integrated.*> embedded", "ICs", "MCU / SoC / FPGA"),
    (r"^integrated.*> memory", "ICs", "memory"),
    (r"^integrated.*> data acquisition", "ICs", "ADC / DAC"),
    (r"^integrated.*> linear", "ICs", "analog"),
    (r"^integrated.*> interface", "ICs", "interface"),
    (r"^integrated.*> logic", "ICs", "logic"),
    (r"^integrated.*> clock", "Crystals & oscillators", "oscillators"),
    (r"^rf and wireless", "ICs", "RF"),
    (r"^integrated", "ICs", None),
    (r"^crystals.*> oscillators", "Crystals & oscillators", "oscillators"),
    (r"^crystals", "Crystals & oscillators", "crystals"),
    (r"^connectors.*usb|hdmi|displayport", "Connectors", "USB & video"),
    (r"^connectors.*coaxial", "Connectors", "RF (coaxial)"),
    (r"^connectors.*(ffc|fpc)", "Connectors", "FPC / FFC"),
    (r"^connectors.*terminal", "Connectors", "terminal blocks"),
    (r"^connectors", "Connectors", "other"),
    (r"^switches", "Buttons & switches", "buttons"),
    (r"test points", "Other", "test points"),
    (r"^sensors", "Other", "sensors"),
]

# ICs by keywords (Description / Name / LibRef of an IC without a usable datasheet category)
IC_RULES = [
    (r"charger|charging", "power: battery charger"),  # before DC-DC: switching chargers say "buck" too
    (r"buck|step[- ]?down|step[- ]?up|boost|dc[- ]?dc|switching regulator", "power: DC-DC"),
    (r"\bldo\b|low[- ]dropout|linear regulator|ams1117|voltage regulator", "power: LDO"),
    (r"power|pmic|load switch|supervisor|reset ic", "power: other"),
    (r"microcontroller|\bmcu\b|\bsoc\b|fpga|stm32|nrf5\d|rk3\d|xc7|cortex|processor", "MCU / SoC / FPGA"),
    (r"flash|eeprom|sram|dram|ddr|memory", "memory"),
    (r"bluetooth|wi-?fi|802\.15|\brf\b|wireless|transceiver module", "RF"),
    (r"\badc\b|\bdac\b|analog[- ]to[- ]digital|digital[- ]to[- ]analog", "ADC / DAC"),
    (r"op[- ]?amp|operational amplifier|amplifier|comparator|current[- ]sense|reference", "analog"),
    (r"shift register|74hc|74lvc|sn74|buffer|inverter|gate|flip[- ]flop|logic", "logic"),
    (r"multiplexer|analog switch|usb|uart|i2c|spi|lvds|level[- ]shift|translator|interface|expander|serializer|hub", "interface"),
]

# Other parts by keywords (LibRef + Name first, then Description): (regex, group, subgroup)
KW_RULES = [
    (r"thermistor|\bntc\b|\bptc\b", "Resistors", "thermistors"),
    (r"resistor[_ ]?array|res[_ ]?array|resistor network", "Resistors", "arrays"),
    (r"resistor|^res\b|\bres_", "Resistors", "chip resistors"),
    (r"capacitor[_ ]?polar|electrolytic|tantal|polymer cap", "Capacitors", "polarized"),
    (r"capacitor|cercap|\bmlcc\b|ceramic cap", "Capacitors", "ceramic"),
    (r"ferrite|\bbead", "Inductors", "ferrite beads"),
    (r"inductor|choke", "Inductors", "chokes"),
    (r"\besd|\btvs|usblc|tpd\de", "Diodes & protection", "ESD / TVS"),  # ESD0402 is often drawn as a zener
    (r"zener", "Diodes & protection", "Zener"),
    (r"schottky|diod|rectifier", "Diodes & protection", "rectifier / Schottky"),
    (r"mosfet|\bfet\b", "Transistors", "MOSFET"),
    (r"\bbjt\b|\bnpn\b|\bpnp\b|transistor", "Transistors", "bipolar"),
    (r"optocoupler|optoisolator|pc817|ltv-?8", "LEDs & optocouplers", "optocouplers"),
    (r"\bled\b|_led|led_|ws28\d\d|rgb", "LEDs & optocouplers", "LEDs"),
    (r"oscillator|\btcxo\b|\bxo\b", "Crystals & oscillators", "oscillators"),
    (r"crystal|xtal|quartz", "Crystals & oscillators", "crystals"),
    (r"button|switch|tact", "Buttons & switches", "buttons"),
    (r"usb|hdmi|displayport", "Connectors", "USB & video"),
    (r"u\.fl|\bsma|coax|\bipex", "Connectors", "RF (coaxial)"),
    (r"\bfpc|\bffc", "Connectors", "FPC / FFC"),
    (r"terminal|2edg|dg30\d|wf-\d", "Connectors", "terminal blocks"),
    (r"micro-match|\bpls|\bpbs|\bidc|header|connector|jumper|socket|\bmics\b", "Connectors", "other"),
    (r"test ?point", "Other", "test points"),
]

PREFIX_RULES = [  # designator prefix -> (group, subgroup)
    (r"^(r|rn|ra)$", "Resistors", "chip resistors"), (r"^c$", "Capacitors", "ceramic"),
    (r"^fb$", "Inductors", "ferrite beads"), (r"^l$", "Inductors", "chokes"),
    (r"^(d|vd)$", "Diodes & protection", "rectifier / Schottky"), (r"^(q|vt)$", "Transistors", "other"),
    (r"^(hl|led)$", "LEDs & optocouplers", "LEDs"), (r"^(u|ic|da|dd)$", "ICs", "other"),
    (r"^(y|zq|bq)$", "Crystals & oscillators", "crystals"), (r"^(j|p|xs|xp|cn|con)$", "Connectors", "other"),
    (r"^(sw|s|sb|btn)$", "Buttons & switches", "buttons"), (r"^tp$", "Other", "test points"),
]

IC_PREFIXES = ("u", "ic", "da", "dd")
NOT_FOR_IC = ("Connectors", "LEDs & optocouplers", "Buttons & switches")  # "USB hub IC" is not a connector


def _first(rules, text):
    for rx, *out in rules:
        m = re.search(rx, text, re.I)
        if m:
            return out, m.group(0)
    return None, None


def classifyLine(it, doc=None):
    """BOM line (+ matched datasheet {category, doc}) -> {group, sub, source, hint}."""
    sym = " ".join(x for x in (it.get("libRef"), it.get("name")) if x)
    desc = it.get("description") or ""
    m = re.match(r"[A-Za-z]+", it["designators"][0]) if it.get("designators") else None
    prefix = m.group(0).lower() if m else ""
    is_ic = prefix in IC_PREFIXES

    def ic_sub(text):
        out, hit = _first(IC_RULES, text)
        return (out[0], hit) if out else (None, None)

    if doc and doc.get("category"):
        cat = " > ".join(doc["category"]).split("#")[0].strip()
        out, _ = _first(DK_RULES, cat)
        if out:
            group, sub = out
            if sub is None:  # PMIC / IC without a usable leaf: the line's keywords pick the subgroup
                sub, _ = ic_sub(f"{sym} {desc} {doc.get('doc', '')}")
                sub = sub or ("power: other" if "pmic" in cat.lower() else "other")
            return {"group": group, "sub": sub, "source": "datasheet", "hint": cat}
    for source, text in (("LibRef/Name", sym), ("description", desc)):
        out, hit = _first(KW_RULES, text)
        if out and not (is_ic and out[0] in NOT_FOR_IC):
            return {"group": out[0], "sub": out[1], "source": source, "hint": hit}
    if is_ic:
        sub, hit = ic_sub(f"{sym} {desc}")
        if sub:
            return {"group": "ICs", "sub": sub, "source": "description", "hint": hit}
        return {"group": "ICs", "sub": "other", "source": "designator", "hint": prefix.upper()}
    out, _ = _first(PREFIX_RULES, prefix)
    if out:
        return {"group": out[0], "sub": out[1], "source": "designator", "hint": prefix.upper()}
    return {"group": "Other", "sub": "unknown", "source": "—", "hint": None}
