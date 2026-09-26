"""Unit tests for the cleaning stage (src/ber/normalize.py, src/ber/lookup.py)."""

from __future__ import annotations

import polars as pl
import pytest

from ber import config
from ber.normalize import clean_frame, clean_text, script_of


def clean_one(name: str = "", address: str = "") -> dict:
    lf = pl.LazyFrame({"business_name": [name], "business_address": [address]}, schema={"business_name": pl.String, "business_address": pl.String})
    return clean_frame(lf).collect().row(0, named=True)


def ct(s: str) -> str:
    return pl.select(clean_text(pl.lit(s))).item()


# --- Indian scripts must survive -------------------------------------------------------------

INDIC = [
    "प्राइवेट लिमिटेड",          # Devanagari
    "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",          # Kannada
    "பிரைவேட் லிமிடெட்",        # Tamil
    "ప్రైవేట్ లిమిటెడ్",          # Telugu
    "প্রাইভেট লিমিটেড",          # Bengali
    "പ്രൈവറ്റ് ലിമിറ്റഡ്",        # Malayalam
    "પ્રાઇવેટ લિમિટેડ",          # Gujarati
    "ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ",           # Gurmukhi
    "ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍",         # Odia
]


@pytest.mark.parametrize("text", INDIC)
def test_indic_text_unchanged(text):
    # vowel signs / viramas are combining marks; the accent step must not touch them
    assert ct(text) == text


def test_zwnj_removed_without_splitting_word():
    assert ct("ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್") == "ಕನ್ಸ್ಟ್ರಕ್ಷನ್"


def test_mixed_script_accent_only_on_latin():
    assert ct("Límited स्वस्तिक") == "limited स्वस्तिक"


@pytest.mark.parametrize("text,script", [
    ("abc", "Latin"), ("प्राइवेट", "Devanagari"), ("லிமிடெட்", "Tamil"), ("ಲಿಮಿಟೆಡ್", "Kannada"),
    ("ఐటీ", "Telugu"), ("লিমিটেড", "Bengali"), ("લિમિટેડ", "Gujarati"), ("ਲਿਮਟਿਡ", "Gurmukhi"),
    ("ലിമിറ്റഡ്", "Malayalam"), ("ଲିମିଟେଡ୍", "Odia"), ("Москва", "Other"), ("123", "None"),
    ("इंटरनेशनल trading प्राइवेट", "Devanagari"),
])
def test_script(text, script):
    assert pl.select(script_of(pl.lit(text))).item() == script


# --- French accents, quotes, digits ----------------------------------------------------------

@pytest.mark.parametrize("raw,clean", [
    ("Lycée Crème", "lycee creme"),
    ("Français Hôtel", "francais hotel"),
    ("ÉCOLE PRIMAIRE PRÔTECTION", "ecole primaire protection"),
    ("6 Avenue des Œillets", "6 avenue des oeillets"),
    ("Kolkata Fincap Holdings Private Límited", "kolkata fincap holdings private limited"),
])
def test_latin_accents_removed(raw, clean):
    assert ct(raw) == clean


@pytest.mark.parametrize("raw,clean", [
    ('"""ehpad Club SAS"', "ehpad club sas"),
    ("Fédération du \"\"ehpad", "federation du ehpad"),
    ("Wynny's Cafe", "wynnys cafe"),
    ("l’Atlantique “x”", "latlantique x"),
    ("D\x1asouza Colony", "dsouza colony"),
    ("Phase Â\x80\x93 1", "phase 1"),
    ("Peopleâ\x80\x99s", "peoples"),
])
def test_quotes_and_mojibake(raw, clean):
    assert ct(raw) == clean


def test_non_ascii_digits():
    assert ct("प्लॉट १२३ ௪௫") == "प्लॉट 123 45"


def test_ampersand_and_null_placeholder():
    assert ct("Tesch & Robinson") == "tesch and robinson"
    assert ct("2221 APPLEDOWN DRIVE, <NULL>, CARY") == "2221 appledown drive cary"


# --- '/' and '-' ----------------------------------------------------------------------------

@pytest.mark.parametrize("raw,clean", [
    ("12/3 Main", "12/3 main"),
    ("4-B Park", "4-b park"),
    ("10-62/1, Gopalapatnam", "10-62/1 gopalapatnam"),
    ("B-824 Tower", "b-824 tower"),
    ("Sector-17", "sector 17"),
    ("Hauts-de-France", "hauts de france"),
    ("KH NO. -570/13", "kh no 570/13"),
    ("House No-345`", "house no 345"),
])
def test_slash_dash(raw, clean):
    assert ct(raw) == clean


def test_addr_numbers():
    r = clean_one(address="12/3, 4-B Park, No. 0040, Sector-17")
    assert r["addr_numbers"] == ["12/3", "4-b", "40", "17"]


def test_dotted_acronyms_merged():
    assert ct("Tesch Carolina L.L.C.") == "tesch carolina llc"
    assert ct("R.K. Puram") == "rk puram"
    assert ct("K R Puram") == "kr puram"


# --- legal suffixes ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,families,core", [
    ("City Estate Private Limited", ["LIMITED", "PRIVATE"], "city estate"),
    ("Rose Biotech Pvt. Ltd.", ["LIMITED", "PRIVATE"], "rose biotech"),
    ("Sharma Traders (P) Ltd", ["LIMITED", "PRIVATE"], "sharma traders"),
    ("Insight Traders Private", ["PRIVATE"], "insight traders"),
    ("Ss Seven Solutions Limited", ["LIMITED"], "ss seven solutions"),
    ("Tata Steel Public Limited", ["PUBLIC"], "tata steel"),
    ("Dr Hari Infotech L.L.P.", ["LLP"], "hari infotech"),
    ("Coriss Caley York, Inc", ["INC"], "coriss caley york"),
    ("Acme Incorporated", ["INC"], "acme"),
    ("Acme Limited Liability Company", ["LLC"], "acme"),
    ("Colonial Industries Corp", ["CORP"], "colonial industries"),
    ("Blue Corporation", ["CORP"], "blue"),
    ("Tesch Carolina L.L.C.", ["LLC"], "tesch carolina"),
    ("Deluca Laboratories P.C.", ["PC"], "deluca laboratories"),
    ("Ram & Co", ["CO"], "ram"),
    ("Ram & Co Pvt Ltd", ["CO", "LIMITED", "PRIVATE"], "ram"),
    ("Sai Company", ["CO"], "sai"),
    ("Nous Jeunes SARL", ["SARL"], "nous jeunes"),
    ("Pessac Sante S.A.S.U.", ["SASU"], "pessac sante"),
    ("Club Nautique SAS", ["SAS"], "club nautique"),
    ("Surf Amis S.A.", ["SA"], "surf amis"),
    ("Reunion Amicale EURL", ["EURL"], "reunion amicale"),
    ("Freres SNC", ["SNC"], "freres"),
    ("E.U.R.L. Freres", ["EURL"], "freres"),
    ("Crestline", [], "crestline"),
    # word order is shuffled in S2/S3: unambiguous forms are removed anywhere (EDA 4)
    ("PVT. ASTOR TRADING LTD.", ["LIMITED", "PRIVATE"], "astor trading"),
    ("LLC Prairie Diana", ["LLC"], "prairie diana"),
    ("Dynamic Investment LLC Services", ["LLC"], "dynamic investment services"),
    ("Art Private Limited Services", ["LIMITED", "PRIVATE"], "art services"),
    ("SAS Club Nautique", ["SAS"], "club nautique"),
    # ambiguous short forms only at the start / end
    ("Maa Co Services", [], "maa co services"),
    ("Coca Cola", [], "coca cola"),
    ("Costco Wholesale", [], "costco wholesale"),
    # generic words are never removed (look-alikes ADD them, EDA 6.5)
    ("Sarthi Trading Services Group", [], "sarthi trading services group"),
    # Indic-script forms
    ("राम प्राइवेट लिमिटेड", ["LIMITED", "PRIVATE"], "राम"),
    ("राम प्रा. लि.", ["LIMITED", "PRIVATE"], "राम"),
    ("ಕೃಷ್ಣ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", ["LIMITED", "PRIVATE"], "ಕೃಷ್ಣ"),
    # a name that is only a legal form keeps it
    ("LLC", ["LLC"], "llc"),
])
def test_legal_forms(name, families, core):
    r = clean_one(name=name)
    assert (r["legal_families"], r["name_core"]) == (families, core)


def test_legacy_suffix_class_kept():
    assert clean_one(name="Rose Biotech Pvt. Ltd.")["legal_suffix_class"] == "PRIVATE_LIMITED"


@pytest.mark.parametrize("name,core", [
    ("SARL Working Union", "working union"),
    ("M/s Rose Biotech Pvt. Ltd.", "rose biotech"),
    ("Smt My Service", "my service"),
    ("Mr Ravi Traders", "ravi traders"),
    ("Shri Salar [Textile] #25772", "salar textile"),
    ("Dr Hari Infotech", "hari infotech"),
    ("Shree Trading", "shree trading"),          # part of real names: never stripped
    ("Ms Priya Boutique", "ms priya boutique"),  # "Ms" without the slash is not Messrs
    ("Trading Mr Kumar", "trading mr kumar"),    # only at the start
    ("Smt", "smt"),                              # never empties a name
])
def test_name_prefixes(name, core):
    assert clean_one(name=name)["name_core"] == core


def test_name_tags_domains_and():
    r = clean_one(name="Golden Tech #67693")
    assert (r["name_core"], r["name_numbers"]) == ("golden tech", ["67693"])
    r = clean_one(name="sarthitrading.com")
    assert (r["name_core"], r["name_domain_stem"], r["name_nospace"]) == ("sarthitrading", "sarthitrading", "sarthitrading")
    assert clean_one(name="www.Sarthi-Trading.co.in")["name_domain_stem"] == "sarthitrading"
    assert clean_one(name="Sarthi @sarthi_trading")["name_domain_stem"] == "sarthitrading"
    assert clean_one(name="Sarthi Trading")["name_domain_stem"] is None
    assert clean_one(name="Sarthi Trading")["name_nospace"] == "sarthitrading"
    assert clean_one(name="Tesch et Robinson")["name_core"] == clean_one(name="Tesch & Robinson")["name_core"]
    assert clean_one(name="Family [Society]")["name_core"] == "family society"


@pytest.mark.parametrize("raw,clean", [
    ("H0uston", "houston"), ("De1hi", "delhi"), ("Ree1sch", "reelsch"), ("H0ust0n", "houston"),
    ("2Nd Floor", "2nd floor"), ("12b", "12b"), ("1-11-251/1B", "1-11-251/1b"), ("S1W33789", "s1w33789"),
    ("Rz-0040", "rz 0040"),
])
def test_ocr_digits(raw, clean):
    assert ct(raw) == clean


@pytest.mark.parametrize("raw,clean", [
    ("Main St, null, Cary", "main st cary"), ("NULL", ""), ("<NULL>", ""), ("N/A, Pune", "pune"),
    ("##3305 Sharatin", "3305 sharatin"), ("3205 # 4 Main", "3205 4 main"), ("Nullah Road", "nullah road"),
])
def test_filler_tokens(raw, clean):
    assert ct(raw) == clean


def test_mojibake_dash_and_script():
    assert ct("WORLDMARK Â\x80\x93 3") == "worldmark 3"
    assert ct("1Â\x80\x932") == "1-2"
    assert clean_one(name="ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ Ltd")["name_script"] == "Kannada"


# --- addresses ---------------------------------------------------------------------------------

@pytest.mark.parametrize("address,std", [
    # st -> street
    ("1224 Wabash St, Frankfort, Indiana", "1224 wabash street frankfort indiana"),
    ("5 Main St Apt 4, Dover", "5 main street apartment 4 dover"),
    ("190 Mesquite St Street, Lordsburg", "190 mesquite street street lordsburg"),
    # st -> saint
    ("1648 Beale St, St Charles, MO", "1648 beale street saint charles mo"),
    ("No 8 Av. Des Acacias, St-Herblain", "8 avenue des acacias saint herblain"),
    ("132, St. Marys Road, Chennai", "132 saint marys road chennai"),
    # ste
    ("25 Liberty Street, Unit STE 110, Harrisonburg", "25 liberty street unit suite 110 harrisonburg"),
    ("29 Boulevard de l'Ocean Ste Marie, Pornic", "29 boulevard de locean sainte marie pornic"),
    # r -> rue only after a number, not before a street type
    ("106 R Neuve, Calais", "106 rue neuve calais"),
    ("231 R St, Washington, DC", "231 r street washington dc"),
    ("C/O R. Kanthasamy, Thottiam", "co r kanthasamy thottiam"),
    # dr: drive at clause end, Doctor elsewhere
    ("354 Branchwood Dr, Evansville", "354 branchwood drive evansville"),
    ("Dr Ambedkar Nagar, New Delhi", "dr ambedkar nagar new delhi"),
    # fl: Florida vs floor
    ("5548 Mount Tabor Road, Greenwood, FL", "5548 mount tabor road greenwood fl"),
    ("8391 Windtree Court, Fl 2, Millersville", "8391 windtree court floor 2 millersville"),
    # bd / av after a number (France), untouched elsewhere
    ("47 Bd du Tertre, Nantes", "47 boulevard du tertre nantes"),
    ("House No-67, Block-BD, Shalimar Bagh", "house 67 block bd shalimar bagh"),
    # always-safe abbreviations
    ("12 Ellis Rd, Tahlequah", "12 ellis road tahlequah"),
    ("First Floor, 12 Park Ave", "1st floor 12 park avenue"),
])
def test_addr_std(address, std):
    assert clean_one(address=address)["addr_std"] == std


def test_landmarks():
    r = clean_one(address="No.510, Opp. Cmh Road, Indira Nagar, Near Devamatha School, Bangalore")
    assert r["addr_landmark"] == "opp cmh road, near devamatha school"
    assert r["addr_no_landmark"] == "no 510 indira nagar bangalore"
    # clause ends at the comma; text before the keyword in the same clause stays
    r = clean_one(address="Amlapukur Road Near Adhunika Mall, Bardhaman")
    assert (r["addr_landmark"], r["addr_no_landmark"]) == ("near adhunika mall", "amlapukur road bardhaman")


@pytest.mark.parametrize("address", ["902 Brantley Street, Opp, AL", "303 Opp Avenue, Andalusia, Alabama", "8744 Landmark Road, VA"])
def test_landmark_false_positives(address):
    r = clean_one(address=address)
    assert r["addr_landmark"] is None
    assert r["addr_no_landmark"] == r["addr_clean"]


def test_raw_columns_unchanged():
    lf = pl.LazyFrame({"business_name": ['"""ehpad Club SAS"'], "business_address": ["106 R. Neuve"], "x": [1]})
    out = clean_frame(lf).collect()
    assert out["business_name"][0] == '"""ehpad Club SAS"'
    assert out["business_address"][0] == "106 R. Neuve"
    assert out["x"][0] == 1


@pytest.mark.parametrize("address", ["", "null", "N/A", " , <NULL> ,", None])
def test_empty_address_is_null(address):
    r = clean_one(name="X", address=address)
    from ber.normalize import STAGE1_COLUMNS
    assert all(r[c] is None for c in STAGE1_COLUMNS if c.startswith("addr_")), r


@pytest.mark.parametrize("address,house,unit", [
    ("3305 Sharatin Road, Kodiak", "3305", None),
    ("AK, ##3305 SHARATIN ROAD, KODIAK", "3305", None),
    ("Unit 16B, 123 Main St, Dover", "123", "16b"),
    ("123 Main St, Fl 0, Dover", "123", "0"),
    ("123 Main St Suite 210", "123", "210"),
    ("Rz-0040, Delhi", "40", None),
    ("03/C Main Road", "3/c", None),
    ("6-3-663/G/4, Punjagutta", "6-3-663/g/4", None),
    ("#153 Park Street", "153", None),
    ("10501 1/2 Elm Dr", "10501", None),
    ("5575-5577 Carter Rd", "5575-5577", None),
    ("8931-C Oak Ave", "8931-c", None),
    ("16654. Main St", "16654", None),
    ("9768- Main St", "9768", None),
    ("87th Street, 2Nd Floor, 12 Park Ave", "12", None),
    ("2Nd Floor, 1-11-251/1B, Hyderabad", "1-11-251/1b", None),
    # India S2/S3 injected leading clause: skipped when another candidate exists
    ("Door No 709 Block No 3 Flat No 3 Krupa", "3", "3"),
    ("Block No 3 Flat No 3 Krupa, Pune", "3", "3"),
    ("Door No 1-65 Dammaiguda, Hyderabad", "1-65", None),
    ("H.no 15, Sector 4, Noida", "4", None),
    ("Plot 48 3Rd Cross Street, Chennai", "48", None),
    ("NO 32, MG Road", "32", None),
    ("Park Street, Kolkata", None, None),
])
def test_house_number(address, house, unit):
    r = clean_one(address=address)
    assert (r["addr_house_number"], r["addr_unit"]) == (house, unit)


@pytest.mark.parametrize("address,state,core", [
    ("1224 Wabash St, Frankfort, Indiana", "indiana", "1224 wabash street frankfort"),
    ("TX, Houston, 12 Main St", "tx", "houston 12 main street"),
    ("80 Woodfield, Rocky Hill, CT", "ct", "80 woodfield rocky hill"),
    ("Plot 5, Andheri, Mumbai, Maharashtra", "maharashtra", "plot 5 andheri mumbai"),
    ("Plot 5, Andheri, Mumbai, MH", "mh", "plot 5 andheri mumbai"),
    ("Plot 5, Andheri, Mumbai, महाराष्ट्र", "महाराष्ट्र", "plot 5 andheri mumbai"),
    ("4 Rue Daurat, Saint-Nazaire, Pays de la Loire", "pays de la loire", "4 rue daurat saint nazaire"),
    ("24 R Desaix, Tourcoing, Nord", "nord", "24 rue desaix tourcoing"),
    ("13 Rue Albert Sauvage, Dunkerque, Hauts-de-France", "hauts de france", "13 rue albert sauvage dunkerque"),
    ("Rz-0040, Palam Colony", None, "rz 40 palam colony"),
    ("सेक्टर 5, नोएडा", None, "सेक्टर 5 नोएडा"),  # all non-Latin: nothing is taken as the state
])
def test_state_slot(address, state, core):
    r = clean_one(address=address)
    assert (r["addr_state_raw"], r["addr_core"]) == (state, core)


@pytest.mark.parametrize("address,std", [
    ("067 Production Ct, Independence, KY", "067 production court independence ky"),
    ("n°24 r de la tranquilite, tourcoing", "24 rue de la tranquilite tourcoing"),
    ("No. 5 Allée des Hêtres, Pornic", "5 allee des hetres pornic"),
    ("R du Brun Pin, Tourcoing", "rue du brun pin tourcoing"),
    ("Roubaix, 2 Imp. Lamartine", "roubaix 2 impasse lamartine"),
    ("4 Rte. de Bordeaux, Lege", "4 route de bordeaux lege"),
    ("74 Ch des Rochelles, Saint-Nazaire", "74 chemin des rochelles saint nazaire"),
    ("13 Pl Sainte Eulalie, Bordeaux", "13 place sainte eulalie bordeaux"),
    ("H.NO 588, SEC. 4, Noida", "h 588 sector 4 noida"),
])
def test_addr_std_more(address, std):
    assert clean_one(address=address)["addr_std"] == std


def test_unseen_text_is_noop():
    # rules keyed on token shapes must pass unknown text through (open set of countries)
    r = clean_one(name="Müller Bäckerei GmbH", address="Hauptstraße 5, 10115 Berlin")
    assert r["name_core"] == "muller backerei gmbh" and r["legal_families"] == []
    assert r["addr_house_number"] == "5" and r["addr_state_raw"] is None


# --- Stage 2 leakage rule ------------------------------------------------------------------------

def test_training_pairs_excludes_val():
    from ber.lookup import training_pairs
    gt = pl.DataFrame({"source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
                       "matched_entity_id": ["S2-1", "S3-1", "S2-2", None]})
    pairs = training_pairs(gt, train_ids=["S1-1", "S1-3"], val_ids=["S1-2"])
    assert set(pairs["s1_id"]) == {"S1-1"}          # val entity and the singleton are gone
    with pytest.raises(ValueError):
        training_pairs(gt, train_ids=["S1-1", "S1-2"], val_ids=["S1-2"])


def test_state_map_learn_and_apply():
    from ber.lookup import apply_state_map, learn_state_map
    aligned = pl.DataFrame({"country": ["US"] * 300 + ["India"] * 300,
                            "s1_state": ["tx"] * 300 + ["maharashtra"] * 300,
                            "other_state": ["texas"] * 150 + ["tx"] * 150 + ["mh"] * 150 + ["महाराष्ट्र"] * 150})
    m = learn_state_map(aligned)
    got = {(r["country"], r["variant"]): r["canonical"] for r in m.iter_rows(named=True)}
    assert got[("US", "texas")] == "tx" and got[("US", "tx")] == "tx"
    assert got[("India", "महाराष्ट्र")] == "maharashtra" and got[("India", "mh")] == "maharashtra"
    lf = pl.LazyFrame({"entity_id": ["a", "b", "c", "d"], "country": ["US", "India", "France", "US"],
                       "addr_state_raw": ["texas", "mh", "nord", None]})
    assert apply_state_map(lf, m).collect()["addr_state_canon"].to_list() == ["tx", "maharashtra", None, None]


@pytest.mark.skipif(not (config.DICTS_DIR / "state_map.parquet").exists(), reason="run scripts/05_learn_tables.py first")
def test_learned_table_records_train_only_provenance():
    import hashlib
    import pyarrow.parquet as pq
    train_sha = hashlib.sha256(config.split_ids_path("train_ids").read_bytes()).hexdigest()
    paths = [config.DICTS_DIR / "state_map.parquet",
             *(config.CLEAN_DIR / f"stage2_{sp}_source{s}.parquet" for sp in config.SPLITS for s in config.SOURCES)]
    for path in paths:
        meta = {k.decode(): v.decode() for k, v in pq.read_metadata(path).metadata.items()}
        assert meta["train_ids_sha256"] == train_sha, path.name
        assert "val_ids never used" in meta["learned_from"], path.name


@pytest.mark.skipif(not (config.DICTS_DIR / "state_map.parquet").exists(), reason="run scripts/05_learn_tables.py first")
def test_france_state_canon_is_null():
    df = pl.scan_parquet(config.CLEAN_DIR / "stage2_test_source1.parquet").join(
        pl.scan_parquet(config.clean_source_path("test", 1)).select("entity_id", "country"), on="entity_id")
    n = df.filter(pl.col("country") == "France").select(pl.col("addr_state_canon").is_not_null().sum()).collect().item()
    assert n == 0
