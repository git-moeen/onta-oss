"""Tests for DefaultNormalizer (cograph_client.resolver.er.normalize)."""

from __future__ import annotations

from cograph_client.resolver.er.normalize import DefaultNormalizer
from cograph_client.resolver.er.types import EntitySignals


N = DefaultNormalizer()


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------


def test_name_lowercases_and_strips_whitespace():
    out = N.normalize(EntitySignals(name="  John Smith  "))
    assert out.name == "john smith"
    assert out.name_tokens == ("john", "smith")


def test_name_strips_diacritics():
    out = N.normalize(EntitySignals(name="José Núñez"))
    assert out.name == "jose nunez"


def test_name_drops_honorifics_and_suffixes():
    out = N.normalize(EntitySignals(name="Dr. John Smith Jr."))
    assert out.name == "john smith"
    assert "dr" not in out.name_tokens
    assert "jr" not in out.name_tokens


def test_name_expands_nickname_on_first_token():
    out = N.normalize(EntitySignals(name="Mike O'Brien"))
    assert out.name == "michael o'brien"
    assert out.name_tokens == ("michael", "o'brien")


def test_name_nickname_does_not_expand_in_later_tokens():
    # "mike" appearing as a surname/middle should not be rewritten — only
    # the first token is treated as a given name.
    out = N.normalize(EntitySignals(name="John Mike"))
    assert out.name == "john mike"


def test_name_none_and_empty_are_none():
    assert N.normalize(EntitySignals(name=None)).name is None
    assert N.normalize(EntitySignals(name="")).name is None
    assert N.normalize(EntitySignals(name="   ")).name is None


def test_name_keeps_hyphen_and_apostrophe():
    out = N.normalize(EntitySignals(name="Mary-Jane O'Hara"))
    assert "mary-jane" in out.name_tokens
    assert "o'hara" in out.name_tokens


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def test_email_gmail_dot_strip_and_plus_tag():
    out = N.normalize(EntitySignals(email="John.Doe+newsletter@Gmail.com"))
    assert out.email == "johndoe@gmail.com"
    assert out.email_local == "johndoe"


def test_email_non_gmail_keeps_dots():
    out = N.normalize(EntitySignals(email="john.doe+tag@example.com"))
    assert out.email == "john.doe@example.com"
    assert out.email_local == "john.doe"


def test_email_no_at_sign_treated_as_local():
    out = N.normalize(EntitySignals(email="someuser"))
    assert out.email == "someuser"
    assert out.email_local == "someuser"


def test_email_googlemail_dot_strip():
    out = N.normalize(EntitySignals(email="j.smith@googlemail.com"))
    assert out.email == "jsmith@googlemail.com"


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------


def test_phone_us_10_digits_gets_plus_1():
    assert N.normalize(EntitySignals(phone="(415) 555-1234")).phone_e164 == "+14155551234"


def test_phone_11_digits_leading_1_gets_plus():
    assert N.normalize(EntitySignals(phone="1-415-555-1234")).phone_e164 == "+14155551234"


def test_phone_already_e164_preserved():
    assert N.normalize(EntitySignals(phone="+447911123456")).phone_e164 == "+447911123456"


def test_phone_empty_is_none():
    assert N.normalize(EntitySignals(phone=None)).phone_e164 is None
    assert N.normalize(EntitySignals(phone="")).phone_e164 is None


# ---------------------------------------------------------------------------
# Address
# ---------------------------------------------------------------------------


def test_address_abbreviates_usps_tokens():
    out = N.normalize(EntitySignals(address="123 Main Street North"))
    assert out.address == "123 main st n"


def test_address_strips_unit_suffix():
    out = N.normalize(EntitySignals(address="500 Oak Avenue Apt 4B"))
    assert out.address == "500 oak ave"
    assert "apt" not in out.address_tokens


def test_address_hash_unit_stripped():
    out = N.normalize(EntitySignals(address="500 Oak Ave #4B"))
    assert "4b" not in (out.address or "")


def test_address_tokens_sorted():
    out = N.normalize(EntitySignals(address="123 Main Boulevard"))
    assert out.address_tokens == tuple(sorted(out.address_tokens))


# ---------------------------------------------------------------------------
# DOB
# ---------------------------------------------------------------------------


def test_dob_iso_passthrough():
    assert N.normalize(EntitySignals(dob="1990-05-12")).dob_iso == "1990-05-12"


def test_dob_us_slash_form():
    # MM/DD/YYYY default for slash
    assert N.normalize(EntitySignals(dob="05/12/1990")).dob_iso == "1990-05-12"


def test_dob_european_fallback_when_month_too_big():
    # 13 isn't a valid month, so DD/MM/YYYY applies
    assert N.normalize(EntitySignals(dob="13/05/1990")).dob_iso == "1990-05-13"


def test_dob_dash_form_prefers_dmy():
    # spec: sentinel "-" -> ISO interpretation; we read this as preferring
    # day-first for ambiguous 3-part dash dates.
    assert N.normalize(EntitySignals(dob="12-05-1990")).dob_iso == "1990-05-12"


def test_dob_unparseable_is_none():
    assert N.normalize(EntitySignals(dob="not a date")).dob_iso is None
    assert N.normalize(EntitySignals(dob=None)).dob_iso is None
    assert N.normalize(EntitySignals(dob="")).dob_iso is None


# ---------------------------------------------------------------------------
# Combined / sanity
# ---------------------------------------------------------------------------


def test_full_signals_roundtrip():
    out = N.normalize(
        EntitySignals(
            name="Dr. Mike O'Brien Jr.",
            email="Mike.OBrien+vip@gmail.com",
            phone="(212) 555-9876",
            address="42 Elm Street Apt 3",
            dob="1985-11-03",
        )
    )
    assert out.name == "michael o'brien"
    assert out.email == "mikeobrien@gmail.com"
    assert out.phone_e164 == "+12125559876"
    assert out.address == "42 elm st"
    assert out.dob_iso == "1985-11-03"
