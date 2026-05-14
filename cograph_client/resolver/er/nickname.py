"""Nickname -> canonical given name mapping for ER name normalization.

Lowercase keys mapping to lowercase canonical given names. Lookup is
one-directional: nickname -> canonical. We do NOT attempt the reverse
(canonical -> all nicknames) because that's a one-to-many relation and
isn't what the normalizer needs.

Coverage is intentionally English-leaning. Non-Latin / transliterated
nicknames (Hiro, Pri, Sasha, Mei, Lena) are included where they have
a well-known canonical equivalent, but this list is not exhaustive
for non-English names.
"""

from __future__ import annotations

NICKNAME_TO_CANONICAL: dict[str, str] = {
    # John / Jonathan
    "jon": "john",
    "johnny": "john",
    "jonny": "john",
    "jack": "john",
    "jackie": "john",
    "jonathan": "john",
    # Michael
    "mike": "michael",
    "mikey": "michael",
    "mick": "michael",
    "micky": "michael",
    "mickey": "michael",
    # James
    "jim": "james",
    "jimmy": "james",
    "jimbo": "james",
    "jamie": "james",
    # Daniel
    "dan": "daniel",
    "danny": "daniel",
    "dani": "daniel",
    # Sarah
    "sara": "sarah",
    "sally": "sarah",
    "sadie": "sarah",
    # Elizabeth
    "liz": "elizabeth",
    "lizzy": "elizabeth",
    "lizzie": "elizabeth",
    "beth": "elizabeth",
    "betsy": "elizabeth",
    "betty": "elizabeth",
    "eliza": "elizabeth",
    "ellie": "elizabeth",
    "libby": "elizabeth",
    # Robert
    "bob": "robert",
    "bobby": "robert",
    "rob": "robert",
    "robby": "robert",
    "robbie": "robert",
    "bert": "robert",
    # William
    "bill": "william",
    "billy": "william",
    "will": "william",
    "willy": "william",
    "willie": "william",
    "liam": "william",
    # Katherine / Catherine
    "kate": "katherine",
    "katie": "katherine",
    "kathy": "katherine",
    "kathie": "katherine",
    "kat": "katherine",
    "kitty": "katherine",
    "katy": "katherine",
    "cathy": "katherine",
    "cate": "katherine",
    # Thomas
    "tom": "thomas",
    "tommy": "thomas",
    "thom": "thomas",
    # Peter
    "pete": "peter",
    "petey": "peter",
    # Joseph
    "joe": "joseph",
    "joey": "joseph",
    # Note: "jose" is the canonical Spanish form, not a nickname — do not map.
    # Stephen / Steven
    "steve": "stephen",
    "stevie": "stephen",
    "steven": "stephen",
    # Christopher
    "chris": "christopher",
    "christie": "christopher",
    "kris": "christopher",
    "topher": "christopher",
    # Nicholas
    "nick": "nicholas",
    "nicky": "nicholas",
    "cole": "nicholas",
    # Matthew
    "matt": "matthew",
    "matty": "matthew",
    # Jennifer
    "jen": "jennifer",
    "jenny": "jennifer",
    "jenn": "jennifer",
    "jennie": "jennifer",
    # Patrick / Patricia
    "pat": "patrick",
    "patty": "patrick",
    "paddy": "patrick",
    "rick_p": "patrick",  # placeholder, unused
    "tricia": "patricia",
    # Samuel / Samantha
    "sam": "samuel",
    "sammy": "samuel",
    "sammie": "samuel",
    # Andrew
    "andy": "andrew",
    "drew": "andrew",
    "andie": "andrew",
    # Richard
    "rick": "richard",
    "ricky": "richard",
    "dick": "richard",
    "rich": "richard",
    "richie": "richard",
    # Anthony
    "tony": "anthony",
    "ant": "anthony",
    # Edward
    "ed": "edward",
    "eddie": "edward",
    "eddy": "edward",
    "ned": "edward",
    "ted_e": "edward",  # avoid collision with theodore
    # Francis
    "frank": "francis",
    "frankie": "francis",
    "fran": "francis",
    # Gregory
    "greg": "gregory",
    "gregg": "gregory",
    # Henry
    "hank": "henry",
    "harry": "henry",
    "hal": "henry",
    # Benjamin
    "ben": "benjamin",
    "benny": "benjamin",
    "benji": "benjamin",
    # Charles
    "charlie": "charles",
    "chuck": "charles",
    "chas": "charles",
    "chip": "charles",
    # Douglas
    "doug": "douglas",
    "dougie": "douglas",
    # Jeffrey
    "jeff": "jeffrey",
    "jeffy": "jeffrey",
    # Kenneth
    "ken": "kenneth",
    "kenny": "kenneth",
    # Lawrence
    "larry": "lawrence",
    "lars": "lawrence",
    # Martin
    "marty": "martin",
    "mart": "martin",
    # Philip / Phillip
    "phil": "philip",
    "phillip": "philip",
    # Raymond
    "ray": "raymond",
    # Ronald
    "ron": "ronald",
    "ronny": "ronald",
    "ronnie": "ronald",
    # Theodore
    "ted": "theodore",
    "teddy": "theodore",
    "theo": "theodore",
    # Walter
    "walt": "walter",
    "wally": "walter",
    # David
    "dave": "david",
    "davey": "david",
    "davy": "david",
    # Donald
    "don": "donald",
    "donny": "donald",
    "donnie": "donald",
    # George
    "georgie": "george",
    # Albert
    "al": "albert",
    "bert_a": "albert",
    "albie": "albert",
    # Alexander / Alexandra
    "alex": "alexander",
    "alec": "alexander",
    "xander": "alexander",
    "sasha": "alexander",
    "sandy": "alexandra",
    "sandra": "alexandra",
    "lexi": "alexandra",
    "lex": "alexander",
    # Elena
    "lena": "elena",
    "ellen": "elena",
    # Meili (CN romanization)
    "mei": "meili",
    # Hiroshi
    "hiro": "hiroshi",
    # Priya
    "pri": "priya",
    # Margaret
    "maggie": "margaret",
    "meg": "margaret",
    "peggy": "margaret",
    "marge": "margaret",
    "margie": "margaret",
    # Victoria
    "vicky": "victoria",
    "tori": "victoria",
    "vic": "victoria",
    # Rebecca
    "becky": "rebecca",
    "becca": "rebecca",
    "reba": "rebecca",
    # Deborah
    "deb": "deborah",
    "debbie": "deborah",
    "debby": "deborah",
    # Barbara
    "barb": "barbara",
    "barbie": "barbara",
    # Susan
    "sue": "susan",
    "susie": "susan",
    "suzy": "susan",
    # Nancy
    "nan": "nancy",
    # Jessica
    "jess": "jessica",
    "jessie": "jessica",
    # Cynthia
    "cindy": "cynthia",
    "cyndi": "cynthia",
    # Christina / Christine
    "chrissy": "christina",
    "tina": "christina",
    "christy": "christina",
    # Stephanie
    "steph": "stephanie",
    "stephie": "stephanie",
    # Vincent
    "vince": "vincent",
    "vinny": "vincent",
    "vin": "vincent",
    # Eugene
    "gene": "eugene",
    # Russell
    "russ": "russell",
    "rusty": "russell",
    # Timothy
    "tim": "timothy",
    "timmy": "timothy",
    # Bernard
    "bernie": "bernard",
    # Frederick
    "fred": "frederick",
    "freddy": "frederick",
    "freddie": "frederick",
    "rick_f": "frederick",
    # Gerald
    "gerry": "gerald",
    "jerry": "gerald",
    # Howard
    "howie": "howard",
    # Joshua
    "josh": "joshua",
    # Nathan / Nathaniel
    "nate": "nathaniel",
    "nat": "nathaniel",
    # Zachary
    "zach": "zachary",
    "zack": "zachary",
    # Isabella / Isabel
    "izzy": "isabella",
    "bella": "isabella",
    "belle": "isabella",
    # Caroline
    "carrie": "caroline",
    "caro": "caroline",
    # Abigail
    "abby": "abigail",
    "abbie": "abigail",
    # Penelope
    "penny": "penelope",
    # Olivia
    "liv": "olivia",
    "livvy": "olivia",
    # Amanda
    "mandy": "amanda",
    # Pamela
    "pam": "pamela",
    # Veronica
    "ronnie_v": "veronica",
    "vera": "veronica",
    # Eleanor
    "nora": "eleanor",
    "ellie_e": "eleanor",
    # Geoffrey
    "geoff": "geoffrey",
    # Maxwell / Maximilian
    "max": "maxwell",
    # Harold
    "harold_h": "harold",
}
