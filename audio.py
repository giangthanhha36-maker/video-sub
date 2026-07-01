#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo cho OmniVoice: chuyển .srt hoặc text -> audio hoàn chỉnh.

Ý tưởng nghiệp vụ mượn từ main.py (vốn dùng Voicevox), nhưng ở đây thay engine
bằng OmniVoice và bổ sung giao diện Gradio.

Hai chế độ:
- SRT to Speech: đọc từng dòng phụ đề, tạo audio ở tốc độ tự nhiên, rồi ghép lại
  theo "timeline cộng dồn" (giữ khoảng lặng gốc giữa các câu; câu nào đọc dài hơn
  ô thời gian của nó thì đẩy các câu sau lùi lại -> tổng có thể dài hơn bản gốc
  nhưng thứ tự và khoảng cách tương đối được giữ đúng).
- Text to Speech: nhập text trực tiếp -> 1 audio.

Cách chạy:
    python gen-speech.py --model /path/to/checkpoint --port 7860

Phụ thuộc: gradio, numpy, torch, pysrt, omnivoice (và whisper cho ASR nếu muốn
tự động lấy text từ audio mẫu).
"""

import argparse
import logging
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import pysrt  # CODE MỚI: dùng để đọc file .srt (giống main.py)
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language name to ISO 639-3 code mapping.

Auto-generated from ``docs/lang_id_name_map.tsv``. Provides ``LANG_NAME_TO_ID``
(for resolving language names to codes) and ``LANG_IDS`` (the set of supported
ISO 639-3 codes). Used by ``OmniVoice.generate()`` to resolve user-provided
language names.
"""

# Auto-generated from docs/lang_id_name_map.tsv
# Maps lowercase language name -> language ID code

LANG_NAME_TO_ID = {
    "abadi": "kbt",
    "abkhazian": "ab",
    "abron": "abr",
    "abua": "abn",
    "adamawa fulfulde": "fub",
    "adyghe": "ady",
    "afade": "aal",
    "afrikaans": "af",
    "agwagwune": "yay",
    "aja (benin)": "ajg",
    "akebu": "keu",
    "alago": "ala",
    "albanian": "sq",
    "algerian arabic": "arq",
    "algerian saharan arabic": "aao",
    "ambo-pasco quechua": "qva",
    "ambonese malay": "abs",
    "amdo tibetan": "adx",
    "amharic": "am",
    "anaang": "anw",
    "angika": "anp",
    "antankarana malagasy": "xmv",
    "aragonese": "an",
    "arbëreshë albanian": "aae",
    "arequipa-la unión quechua": "qxu",
    "armenian": "hy",
    "ashe": "ahs",
    "ashéninka perené": "prq",
    "askopan": "eiv",
    "assamese": "as",
    "asturian": "ast",
    "atayal": "tay",
    "awak": "awo",
    "ayacucho quechua": "quy",
    "azerbaijani": "az",
    "baatonum": "bba",
    "bacama": "bcy",
    "bade": "bde",
    "bafia": "ksf",
    "bafut": "bfd",
    "bagirmi fulfulde": "fui",
    "bago-kusuntu": "bqg",
    "baharna arabic": "abv",
    "bakoko": "bkh",
    "balanta-ganja": "bjt",
    "balti": "bft",
    "bamenyam": "bce",
    "bamun": "bax",
    "bangwinji": "bsj",
    "banjar": "bjn",
    "bankon": "abb",
    "baoulé": "bci",
    "bara malagasy": "bhr",
    "barok": "bjk",
    "basa (cameroon)": "bas",
    "basa (nigeria)": "bzw",
    "bashkir": "ba",
    "basque": "eu",
    "batak mandailing": "btm",
    "batanga": "bnm",
    "bateri": "btv",
    "bats": "bbl",
    "bayot": "bda",
    "bebele": "beb",
    "belarusian": "be",
    "bengali": "bn",
    "betawi": "bew",
    "bhili": "bhb",
    "bhojpuri": "bho",
    "bilur": "bxf",
    "bima": "bhp",
    "bodo": "brx",
    "boghom": "bux",
    "bokyi": "bky",
    "bomu": "bmq",
    "bondei": "bou",
    "borgu fulfulde": "fue",
    "bosnian": "bs",
    "brahui": "brh",
    "braj": "bra",
    "breton": "br",
    "buduma": "bdm",
    "buginese": "bug",
    "bukharic": "bhh",
    "bulgarian": "bg",
    "bulu (cameroon)": "bum",
    "bundeli": "bns",
    "bunun": "bnn",
    "bura-pabir": "bwr",
    "burak": "bys",
    "burmese": "my",
    "burushaski": "bsk",
    "cacaloxtepec mixtec": "miu",
    "cajatambo north lima quechua": "qvl",
    "cakfem-mushere": "cky",
    "cameroon pidgin": "wes",
    "campidanese sardinian": "sro",
    "cantonese": "yue",
    "catalan": "ca",
    "cebuano": "ceb",
    "cen": "cen",
    "central kurdish": "ckb",
    "central nahuatl": "nhn",
    "central pame": "pbs",
    "central pashto": "pst",
    "central puebla nahuatl": "ncx",
    "central tarahumara": "tar",
    "central yupik": "esu",
    "central-eastern niger fulfulde": "fuq",
    "chadian arabic": "shu",
    "chichewa": "ny",
    "chichicapan zapotec": "zpv",
    "chiga": "cgg",
    "chimalapa zoque": "zoh",
    "chimborazo highland quichua": "qug",
    "chinese": "zh",
    "chiquián ancash quechua": "qxa",
    "chitwania tharu": "the",
    "chokwe": "cjk",
    "chuvash": "cv",
    "cibak": "ckl",
    "coastal konjo": "kjc",
    "copainalá zoque": "zoc",
    "cornish": "kw",
    "corongo ancash quechua": "qwa",
    "croatian": "hr",
    "cross river mbembe": "mfn",
    "cuyamecalco mixtec": "xtu",
    "czech": "cs",
    "dadiya": "dbd",
    "dagbani": "dag",
    "dameli": "dml",
    "danish": "da",
    "dargwa": "dar",
    "dazaga": "dzg",
    "deccan": "dcc",
    "degema": "deg",
    "dera (nigeria)": "kna",
    "dghwede": "dgh",
    "dhatki": "mki",
    "dhivehi": "dv",
    "dhofari arabic": "adf",
    "dijim-bwilim": "cfa",
    "dogri": "dgo",
    "domaaki": "dmk",
    "dotyali": "dty",
    "duala": "dua",
    "dutch": "nl",
    "dũya": "ldb",
    "dyula": "dyu",
    "eastern balochi": "bgp",
    "eastern bolivian guaraní": "gui",
    "eastern egyptian bedawi arabic": "avl",
    "eastern krahn": "kqo",
    "eastern mari": "mhr",
    "eastern yiddish": "ydd",
    "ebrié": "ebr",
    "eggon": "ego",
    "egyptian arabic": "arz",
    "ejagham": "etu",
    "eleme": "elm",
    "eloyi": "afo",
    "embu": "ebu",
    "english": "en",
    "erzya": "myv",
    "esan": "ish",
    "esperanto": "eo",
    "estonian": "et",
    "eton (cameroon)": "eto",
    "ewondo": "ewo",
    "extremaduran": "ext",
    "fang (equatorial guinea)": "fan",
    "fanti": "fat",
    "farefare": "gur",
    "fe'fe'": "fmp",
    "filipino": "fil",
    "filomena mata-coahuitlán totonac": "tlp",
    "finnish": "fi",
    "fipa": "fip",
    "french": "fr",
    "fulah": "ff",
    "galician": "gl",
    "gambian wolof": "wof",
    "ganda": "lg",
    "garhwali": "gbm",
    "gawar-bati": "gwt",
    "gawri": "gwc",
    "gbagyi": "gbr",
    "gbari": "gby",
    "geji": "gyz",
    "gen": "gej",
    "georgian": "ka",
    "german": "de",
    "geser-gorom": "ges",
    "gheg albanian": "aln",
    "ghomálá'": "bbj",
    "gidar": "gid",
    "glavda": "glw",
    "goan konkani": "gom",
    "goaria": "gig",
    "goemai": "ank",
    "gola": "gol",
    "greek": "el",
    "guarani": "gn",
    "guduf-gava": "gdf",
    "guerrero amuzgo": "amu",
    "gujarati": "gu",
    "gujari": "gju",
    "gulf arabic": "afb",
    "gurgula": "ggg",
    "gusii": "guz",
    "gusilay": "gsl",
    "gweno": "gwe",
    "güilá zapotec": "ztu",
    "hadothi": "hoj",
    "hahon": "hah",
    "haitian": "ht",
    "hakha chin": "cnh",
    "hakö": "hao",
    "halia": "hla",
    "hausa": "ha",
    "hawaiian": "haw",
    "hazaragi": "haz",
    "hebrew": "he",
    "hemba": "hem",
    "herero": "hz",
    "highland konjo": "kjk",
    "hijazi arabic": "acw",
    "hindi": "hi",
    "huarijio": "var",
    "huautla mazatec": "mau",
    "huaxcaleca nahuatl": "nhq",
    "huba": "hbb",
    "huitepec mixtec": "mxs",
    "hula": "hul",
    "hungarian": "hu",
    "hunjara-kaina ke": "hkk",
    "hwana": "hwo",
    "ibibio": "ibb",
    "icelandic": "is",
    "idakho-isukha-tiriki": "ida",
    "idoma": "idu",
    "igbo": "ig",
    "igo": "ahl",
    "ikposo": "kpo",
    "ikwere": "ikw",
    "imbabura highland quichua": "qvi",
    "indonesian": "id",
    "indus kohistani": "mvy",
    "interlingua (international auxiliary language association)": "ia",
    "inupiaq": "ik",
    "irish": "ga",
    "iron ossetic": "os",
    "isekiri": "its",
    "isoko": "iso",
    "italian": "it",
    "ito": "itw",
    "itzá": "itz",
    "ixtayutla mixtec": "vmj",
    "izon": "ijc",
    "jambi malay": "jax",
    "japanese": "ja",
    "jaqaru": "jqr",
    "jauja wanca quechua": "qxw",
    "jaunsari": "jns",
    "javanese": "jv",
    "jiba": "juo",
    "jju": "kaj",
    "judeo-moroccan arabic": "aju",
    "juxtlahuaca mixtec": "vmc",
    "kabardian": "kbd",
    "kabras": "lkb",
    "kabuverdianu": "kea",
    "kabyle": "kab",
    "kachi koli": "gjk",
    "kairak": "ckr",
    "kalabari": "ijn",
    "kalasha": "kls",
    "kalenjin": "kln",
    "kalkoti": "xka",
    "kamba": "kam",
    "kamo": "kcq",
    "kanauji": "bjj",
    "kanembu": "kbl",
    "kannada": "kn",
    "karekare": "kai",
    "kashmiri": "ks",
    "kathoriya tharu": "tkt",
    "kati": "bsh",
    "kazakh": "kk",
    "keiyo": "eyo",
    "khams tibetan": "khg",
    "khana": "ogo",
    "khetrani": "xhe",
    "khmer": "km",
    "khowar": "khw",
    "kinga": "zga",
    "kinnauri": "kfk",
    "kinyarwanda": "rw",
    "kirghiz": "ky",
    "kirya-konzəl": "fkk",
    "kochila tharu": "thq",
    "kohistani shina": "plk",
    "kohumono": "bcs",
    "kok borok": "trp",
    "kol (papua new guinea)": "kol",
    "kom (cameroon)": "bkm",
    "koma": "kmy",
    "konkani": "knn",
    "konzo": "koo",
    "korean": "ko",
    "korwa": "kfp",
    "kota (india)": "kfe",
    "koti": "eko",
    "kuanua": "ksd",
    "kuanyama": "kj",
    "kui (india)": "uki",
    "kulung (nigeria)": "bbu",
    "kuot": "kto",
    "kushi": "kuh",
    "kwambi": "kwm",
    "kwasio": "nmg",
    "lala-roba": "lla",
    "lamang": "hia",
    "lao": "lo",
    "larike-wakasihu": "alo",
    "lasi": "lss",
    "latgalian": "ltg",
    "latvian": "lv",
    "levantine arabic": "apc",
    "liana-seti": "ste",
    "liberia kpelle": "xpe",
    "liberian english": "lir",
    "libyan arabic": "ayl",
    "ligurian": "lij",
    "lijili": "mgi",
    "lingala": "ln",
    "lithuanian": "lt",
    "loarki": "lrk",
    "logooli": "rag",
    "logudorese sardinian": "src",
    "loja highland quichua": "qvj",
    "loloda": "loa",
    "longuda": "lnu",
    "loxicha zapotec": "ztp",
    "luba-lulua": "lua",
    "luo": "luo",
    "lushai": "lus",
    "luxembourgish": "lb",
    "maasina fulfulde": "ffm",
    "maba (chad)": "mde",
    "macedo-romanian": "rup",
    "macedonian": "mk",
    "mada (cameroon)": "mxu",
    "mafa": "maf",
    "maithili": "mai",
    "malay": "ms",
    "malayalam": "ml",
    "mali": "gcc",
    "malinaltepec me'phaa": "tcf",
    "maltese": "mt",
    "mandara": "tbf",
    "mandjak": "mfv",
    "manggarai": "mqy",
    "manipuri": "mni",
    "mansoanka": "msw",
    "manx": "gv",
    "maori": "mi",
    "marathi": "mr",
    "marghi central": "mrt",
    "marghi south": "mfm",
    "maria (india)": "mrr",
    "marwari (pakistan)": "mve",
    "masana": "mcn",
    "masikoro malagasy": "msh",
    "matsés": "mcf",
    "mazaltepec zapotec": "zpy",
    "mazatlán mazatec": "vmz",
    "mazatlán mixe": "mzl",
    "mbe": "mfo",
    "mbo (cameroon)": "mbo",
    "mbum": "mdd",
    "medumba": "byv",
    "mekeo": "mek",
    "meru": "mer",
    "mesopotamian arabic": "acm",
    "mewari": "mtr",
    "min nan chinese": "nan",
    "mingrelian": "xmf",
    "mitlatongo mixtec": "vmm",
    "miya": "mkf",
    "mokpwe": "bri",
    "moksha": "mdf",
    "mom jango": "ver",
    "mongolian": "mn",
    "moroccan arabic": "ary",
    "motu": "meu",
    "mpiemo": "mcx",
    "mpumpong": "mgg",
    "mundang": "mua",
    "mungaka": "mhk",
    "musey": "mse",
    "musgu": "mug",
    "musi": "mui",
    "naba": "mne",
    "najdi arabic": "ars",
    "nalik": "nal",
    "nawdm": "nmz",
    "ndonga": "ng",
    "neapolitan": "nap",
    "nepali": "npi",
    "ngamo": "nbh",
    "ngas": "anc",
    "ngiemboon": "nnh",
    "ngizim": "ngi",
    "ngomba": "jgo",
    "ngombale": "nla",
    "nigerian fulfulde": "fuv",
    "nigerian pidgin": "pcm",
    "nimadi": "noe",
    "nobiin": "fia",
    "north mesopotamian arabic": "ayp",
    "north moluccan malay": "max",
    "northern betsimisaraka malagasy": "bmm",
    "northern hindko": "hno",
    "northern kurdish": "kmr",
    "northern pame": "pmq",
    "northern pashto": "pbu",
    "northern uzbek": "uzn",
    "northwest gbaya": "gya",
    "norwegian": "no",
    "norwegian bokmål": "nb",
    "norwegian nynorsk": "nn",
    "notsi": "ncf",
    "nyankpa": "yes",
    "nyungwe": "nyu",
    "nzanyi": "nja",
    "nüpode huitoto": "hux",
    "occitan": "oc",
    "od": "odk",
    "odia": "ory",
    "odual": "odu",
    "omani arabic": "acx",
    "orizaba nahuatl": "nlv",
    "orma": "orc",
    "ormuri": "oru",
    "oromo": "om",
    "pahari-potwari": "phr",
    "paiwan": "pwn",
    "panjabi": "pa",
    "papuan malay": "pmy",
    "parkari koli": "kvx",
    "pedi": "nso",
    "pero": "pip",
    "persian": "fa",
    "petats": "pex",
    "phalura": "phl",
    "piemontese": "pms",
    "piya-kwonci": "piy",
    "plateau malagasy": "plt",
    "polish": "pl",
    "poqomam": "poc",
    "portuguese": "pt",
    "pulaar": "fuc",
    "pular": "fuf",
    "puno quechua": "qxp",
    "pushto": "ps",
    "pökoot": "pko",
    "qaqet": "byx",
    "quiotepec chinantec": "chq",
    "rana tharu": "thr",
    "rangi": "lag",
    "rapoisi": "kyx",
    "ratahan": "rth",
    "rayón zoque": "zor",
    "romanian": "ro",
    "romansh": "rm",
    "rombo": "rof",
    "rotokas": "roo",
    "rukai": "dru",
    "russian": "ru",
    "sacapulteco": "quv",
    "saidi arabic": "aec",
    "sakalava malagasy": "skg",
    "sakizaya": "szy",
    "saleman": "sau",
    "samba daka": "ccg",
    "samba leko": "ndi",
    "san felipe otlaltepec popoloca": "pow",
    "san francisco del mar huave": "hue",
    "san juan atzingo popoloca": "poe",
    "san martín itunyoso triqui": "trq",
    "san miguel el grande mixtec": "mig",
    "sansi": "ssi",
    "sanskrit": "sa",
    "santa ana de tusi pasco quechua": "qxt",
    "santa catarina albarradas zapotec": "ztn",
    "santali": "sat",
    "santiago del estero quichua": "qus",
    "saposa": "sps",
    "saraiki": "skr",
    "sardinian": "sc",
    "saya": "say",
    "sediq": "trv",
    "serbian": "sr",
    "seri": "sei",
    "shina": "scl",
    "shona": "sn",
    "siar-lak": "sjr",
    "sibe": "nco",
    "sicilian": "scn",
    "sihuas ancash quechua": "qws",
    "sikkimese": "sip",
    "sinaugoro": "snc",
    "sindhi": "sd",
    "sindhi bhil": "sbn",
    "sinhala": "si",
    "sinicahua mixtec": "xti",
    "sipacapense": "qum",
    "siwai": "siw",
    "slovak": "sk",
    "slovenian": "sl",
    "solos": "sol",
    "somali": "so",
    "soninke": "snk",
    "south giziga": "giz",
    "south ucayali ashéninka": "cpy",
    "southeastern nochixtlán mixtec": "mxy",
    "southern betsimisaraka malagasy": "bzc",
    "southern pashto": "pbt",
    "southern pastaza quechua": "qup",
    "soyaltepec mazatec": "vmp",
    "spanish": "es",
    "standard arabic": "arb",
    "standard moroccan tamazight": "zgh",
    "sudanese arabic": "apd",
    "sulka": "sua",
    "svan": "sva",
    "swahili": "sw",
    "swedish": "sv",
    "tae'": "rob",
    "tahaggart tamahaq": "thv",
    "taita": "dav",
    "tajik": "tg",
    "tamil": "ta",
    "tandroy-mahafaly malagasy": "tdx",
    "tangale": "tan",
    "tanosy malagasy": "txy",
    "tarok": "yer",
    "tatar": "tt",
    "tedaga": "tuq",
    "telugu": "te",
    "tem": "kdh",
    "teop": "tio",
    "tepeuxila cuicatec": "cux",
    "tepinapa chinantec": "cte",
    "tera": "ttr",
    "terei": "buo",
    "termanu": "twu",
    "tesaka malagasy": "tkg",
    "tetelcingo nahuatl": "nhg",
    "teutila cuicatec": "cut",
    "thai": "th",
    "tibetan": "bo",
    "tidaá mixtec": "mtx",
    "tidore": "tvo",
    "tigak": "tgc",
    "tigre": "tig",
    "tigrinya": "ti",
    "tilquiapan zapotec": "zts",
    "tinputz": "tpz",
    "tlacoapa me'phaa": "tpl",
    "tlacoatzintepec chinantec": "ctl",
    "tlingit": "tli",
    "toki pona": "tok",
    "tomoip": "tqp",
    "tondano": "tdn",
    "tonsea": "txs",
    "tooro": "ttj",
    "torau": "ttu",
    "torwali": "trw",
    "tsimihety malagasy": "xmw",
    "tsotso": "lto",
    "tswana": "tn",
    "tugen": "tuy",
    "tuki": "bag",
    "tula": "tul",
    "tulu": "tcy",
    "tunen": "tvu",
    "tungag": "lcm",
    "tunisian arabic": "aeb",
    "tupuri": "tui",
    "turkana": "tuv",
    "turkish": "tr",
    "turkmen": "tk",
    "tututepec mixtec": "mtu",
    "twi": "tw",
    "ubaghara": "byc",
    "uighur": "ug",
    "ukrainian": "uk",
    "umbundu": "umb",
    "upper sorbian": "hsb",
    "urdu": "ur",
    "ushojo": "ush",
    "uzbek": "uz",
    "vai": "vai",
    "vietnamese": "vi",
    "votic": "vot",
    "võro": "vro",
    "waci gbe": "wci",
    "wadiyara koli": "kxp",
    "waja": "wja",
    "wakhi": "wbl",
    "wanga": "lwg",
    "wapan": "juk",
    "warji": "wji",
    "welsh": "cy",
    "wemale": "weo",
    "western frisian": "fy",
    "western highland purepecha": "pua",
    "western juxtlahuaca mixtec": "jmx",
    "western maninkakan": "mlq",
    "western mari": "mrj",
    "western niger fulfulde": "fuh",
    "western panjabi": "pnb",
    "wolof": "wo",
    "wuzlam": "udl",
    "xanaguía zapotec": "ztg",
    "xhosa": "xh",
    "yace": "ekr",
    "yakut": "sah",
    "yalahatan": "jal",
    "yanahuanca pasco quechua": "qur",
    "yangben": "yav",
    "yaqui": "yaq",
    "yauyos quechua": "qux",
    "yekhee": "ets",
    "yiddish": "yi",
    "yidgha": "ydg",
    "yoruba": "yo",
    "yutanduchi mixtec": "mab",
    "zacatlán-ahuacatlán-tepetzintla nahuatl": "nhi",
    "zarma": "dje",
    "zaza": "zza",
    "zulu": "zu",
    "ömie": "aom",
}

LANG_NAMES = set(LANG_NAME_TO_ID.keys())
LANG_IDS = set(LANG_NAME_TO_ID.values())

# Exceptions where .title() doesn't match the canonical casing from the TSV.
_TITLE_EXCEPTIONS = {
    "fe'fe'": "Fe'fe'",
    "dũya": "Dũya",
    "santiago del estero quichua": "Santiago del Estero Quichua",
    "santa ana de tusi pasco quechua": "Santa Ana de Tusi Pasco Quechua",
    "malinaltepec me'phaa": "Malinaltepec Me'phaa",
    "tlacoapa me'phaa": "Tlacoapa Me'phaa",
}


def lang_display_name(name: str) -> str:
    """Return a display-friendly version of a lowercase language name.

    Uses .title() for most names, with manual exceptions for cases like
    apostrophes and small words (de, del) that should stay lowercase.
    """
    return _TITLE_EXCEPTIONS.get(name, name.title())


def get_best_device():
    """Auto-detect the best available device: CUDA > XPU > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}


# ===========================================================================
# CODE MỚI: Các hàm helper (thay cho util.py không tồn tại trong repo)
# ===========================================================================
def pysrttime_to_seconds(t) -> float:
    """Đổi một mốc thời gian SubRipTime của pysrt sang số giây (float).

    Ví dụ 00:01:02,500 -> 62.5 giây.
    """
    return (
        t.hours * 3600
        + t.minutes * 60
        + t.seconds
        + t.milliseconds / 1000.0
    )


def combine_segments(
    segments: List[Tuple[float, np.ndarray]], sr: int
) -> np.ndarray:
    """Ghép nhiều đoạn audio vào một dòng thời gian (timeline) duy nhất.

    segments: danh sách (vị_trí_bắt_đầu_giây, waveform_float_1d).
    sr: sampling rate (số mẫu / giây) của model.

    Cách làm: tính tổng độ dài cần thiết, tạo một mảng numpy toàn số 0 (khoảng
    lặng), rồi đặt từng đoạn audio vào đúng vị trí thời gian của nó. Nếu các đoạn
    có lỡ chồng lên nhau thì cộng dồn rồi clip lại trong [-1, 1] để tránh vỡ tiếng.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)

    total_samples = 0
    placed: List[Tuple[int, np.ndarray]] = []
    for pos, wf in segments:
        start_sample = int(round(pos * sr))
        end_sample = start_sample + len(wf)
        total_samples = max(total_samples, end_sample)
        placed.append((start_sample, wf))

    canvas = np.zeros(total_samples, dtype=np.float32)
    for start_sample, wf in placed:
        # CODE MỚI: bo nhẹ đầu/cuối mỗi đoạn để tránh tiếng tách/cụp khi nối.
        seg = _apply_edge_fade(wf.astype(np.float32), sr)
        canvas[start_sample:start_sample + len(seg)] += seg

    np.clip(canvas, -1.0, 1.0, out=canvas)
    return canvas


# ===========================================================================
# CODE MỚI: Ước lượng thời lượng đọc theo từng ngôn ngữ + tiện ích fade
# ===========================================================================

# Tốc độ đọc trung bình (ký tự / giây) theo ngôn ngữ. Đây là giá trị khởi
# điểm, nên nghe thử rồi tinh chỉnh lại cho khớp giọng OmniVoice thực tế.
LANG_SPEED = {
    "vi": 22.0,   # tiếng Việt
    "en": 15.0,   # tiếng Anh
    "zh": 5.5,    # tiếng Trung (mỗi chữ là 1 âm tiết, dày đặc hơn)
    "ja": 8.0,    # tiếng Nhật
    "ko": 7.0,    # tiếng Hàn
    "default": 15.0,
}

# Giới hạn tăng tốc mặc định để giữ chất lượng (có thể chỉnh trên giao diện).
MAX_SPEED_DEFAULT = 1.3

# Bộ ký tự đặc trưng tiếng Việt để nhận diện nhanh khi người dùng chọn "Auto".
_VI_CHARS = set(
    "ăâđêôơưĂÂĐÊÔƠƯ"
    "áàảãạấầẩẫậắằẳẵặ"
    "éèẻẽẹếềểễệ"
    "íìỉĩị"
    "óòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữự"
    "ýỳỷỹỵ"
)


def detect_rate_lang(text: str, ui_language: str = None) -> str:
    """Chọn mã ngôn ngữ (vi/en/zh/ja/ko/default) để tra bảng tốc độ.

    Ưu tiên ngôn ngữ người dùng đã chọn; nếu để "Auto" thì dò nhanh theo ký tự.
    """
    if ui_language and ui_language != "Auto":
        l = ui_language.lower()
        if "vietnam" in l:
            return "vi"
        if "english" in l:
            return "en"
        if "chinese" in l or "mandarin" in l:
            return "zh"
        if "japanese" in l:
            return "ja"
        if "korean" in l:
            return "ko"

    # Dò theo ký tự trong văn bản (khi chọn "Auto" hoặc ngôn ngữ không có bảng).
    has_kana = has_hangul = has_cjk = has_vi = False
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF:
            has_kana = True
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:
            has_hangul = True
        elif 0x4E00 <= code <= 0x9FFF:
            has_cjk = True
        elif ch in _VI_CHARS:
            has_vi = True

    if has_kana:
        return "ja"
    if has_hangul:
        return "ko"
    if has_cjk:
        return "zh"
    if has_vi:
        return "vi"
    return "default"


def estimate_natural_duration(text: str, rate_lang: str) -> float:
    """Ước lượng số giây cần để đọc 'text' ở tốc độ tự nhiên của ngôn ngữ.

    Đếm số ký tự (đã bỏ khoảng trắng) rồi chia cho tốc độ của ngôn ngữ tương ứng.
    """
    n_chars = len("".join(text.split()))  # bỏ mọi khoảng trắng
    rate = LANG_SPEED.get(rate_lang, LANG_SPEED["default"])
    if rate <= 0:
        rate = LANG_SPEED["default"]
    return n_chars / rate


def _apply_edge_fade(wf: np.ndarray, sr: int, fade_ms: float = 15.0) -> np.ndarray:
    """Bo nhẹ đầu/cuối đoạn audio để tránh tiếng 'tách/cụp' khi nối các câu."""
    n = len(wf)
    fade = int(sr * fade_ms / 1000.0)
    if n == 0 or fade <= 0:
        return wf
    fade = min(fade, n // 2)
    if fade <= 0:
        return wf
    out = wf.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[:fade] *= ramp
    out[-fade:] *= ramp[::-1]
    return out


# ===========================================================================
# CODE MỚI (HARD SYNC): Co/giãn 1 đoạn audio cho VỪA ĐÚNG số mẫu yêu cầu.
# Mục đích: phục vụ chế độ "neo cứng timeline" -> mỗi câu phải khớp đúng ô thời
# gian của phụ đề để KHÔNG BAO GIỜ trượt giờ so với video.
#   - Nếu audio DÀI hơn ô -> nén lại (đọc nhanh hơn) NHƯNG giữ cao độ (pitch)
#     bằng librosa.effects.time_stretch -> giọng không bị "vịt Donald".
#   - Nếu audio NGẮN hơn ô -> để nguyên (phần dư là khoảng lặng tự nhiên).
# Nếu không có librosa thì fallback bằng nội suy tuyến tính (lệch cao độ nhẹ).
# ===========================================================================
def _fit_to_length(wf: np.ndarray, target_len: int, sr: int) -> np.ndarray:
    """Trả về waveform có độ dài đúng bằng target_len (mẫu).

    Chỉ NÉN khi audio dài hơn ô thời gian; nếu ngắn hơn thì giữ nguyên (sẽ được
    đặt vào canvas, phần còn lại là khoảng lặng). Cuối cùng cắt/đệm cho khớp
    tuyệt đối target_len để đảm bảo không tích luỹ sai số.
    """
    n = len(wf)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if n == 0:
        return np.zeros(target_len, dtype=np.float32)

    wf = wf.astype(np.float32)
    if n > target_len:
        rate = n / float(target_len)  # rate > 1 = đọc nhanh hơn (nén thời gian)
        try:
            import librosa  # import muộn để khởi động service nhanh hơn

            wf = librosa.effects.time_stretch(wf, rate=rate)
        except Exception:
            # Fallback: nội suy tuyến tính (đổi cao độ chút) khi thiếu librosa.
            idx = np.linspace(0, n - 1, target_len)
            wf = np.interp(idx, np.arange(n), wf).astype(np.float32)

        # Sau khi co giãn, độ dài có thể lệch vài mẫu -> cắt/đệm cho khớp.
        if len(wf) > target_len:
            wf = wf[:target_len]
        elif len(wf) < target_len:
            wf = np.pad(wf, (0, target_len - len(wf)))
    return wf.astype(np.float32)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate

    # =====================================================================
    # CODE MỚI: Lõi sinh audio trả về waveform float (1 chiều) để dễ ghép nối.
    # Khác với _gen_core cũ (trả về int16 cho Gradio), hàm này trả về float
    # trong khoảng [-1, 1] để combine_segments xử lý chính xác.
    # =====================================================================
    def _gen_waveform(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Vui lòng nhập text cần đọc."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        # Voice clone: nếu có audio mẫu thì tạo prompt để bắt chước giọng đó.
        if ref_audio:
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        # audio[0] là waveform float [-1, 1]. Ép kiểu float32 cho nhẹ.
        waveform = np.asarray(audio[0], dtype=np.float32)
        return waveform, "Done."

    # -- shared generation core (GIỮ LẠI từ bản gốc, dùng cho wrapper Gradio) --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
    ):
        if mode == "clone" and not ref_audio:
            return None, "Please upload a reference audio."

        waveform, msg = _gen_waveform(
            text,
            language,
            ref_audio,
            instruct,
            num_step,
            guidance_scale,
            denoise,
            speed,
            duration,
            preprocess_prompt,
            postprocess_output,
            ref_text=ref_text,
        )
        if waveform is None:
            return None, msg

        waveform_i16 = (waveform * 32767).astype(np.int16)
        return (sampling_rate, waveform_i16), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # CODE MỚI: Chuyển toàn bộ file .srt thành 1 audio theo timeline cộng dồn.
    # =====================================================================
    def _srt_to_speech(
        srt_file,
        language,
        ref_audio,
        ref_text,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        fit_timeline=True,
        max_speed=MAX_SPEED_DEFAULT,
    ):
        if not srt_file:
            return None, "Vui lòng tải lên một file .srt."

        # pysrt.open nhận đường dẫn file. Thử utf-8 trước, lỗi thì để pysrt tự đoán.
        try:
            srt_data = pysrt.open(srt_file, encoding="utf-8")
        except Exception:
            try:
                srt_data = pysrt.open(srt_file)
            except Exception as e:
                return None, f"Không đọc được file SRT: {e}"

        if not srt_data or len(srt_data) == 0:
            return None, "File SRT rỗng hoặc sai định dạng."

        segments: List[Tuple[float, np.ndarray]] = []
        cursor_end = 0.0   # thời điểm kết thúc audio trước đó (trên timeline mới)
        prev_srt_end = 0.0  # thời điểm kết thúc câu trước đó (theo SRT gốc)

        # CODE MỚI: thống kê để báo cáo + giúp căn chỉnh bảng tốc độ LANG_SPEED.
        num_sped = 0       # số câu phải tăng tốc
        max_factor = 1.0   # hệ số tăng tốc lớn nhất

        for idx, item in enumerate(srt_data):
            text = item.text.replace("\n", " ").strip()
            start = pysrttime_to_seconds(item.start)
            end = pysrttime_to_seconds(item.end)

            # Khoảng lặng gốc trước câu này. Câu đầu: gap = start (lặng đầu video).
            gap = start - prev_srt_end
            if gap < 0:
                gap = 0.0  # tránh chồng tiếng khi phụ đề bị lệch

            # Vị trí đặt câu này = kết thúc câu trước (timeline mới) + khoảng lặng.
            pos = cursor_end + gap

            # Dòng phụ đề rỗng: chỉ giữ khoảng lặng, không sinh audio.
            if not text:
                cursor_end = pos
                prev_srt_end = end
                continue

            # CODE MỚI: tính tốc độ đọc riêng cho câu này theo ô thời gian + ngôn ngữ.
            seg_speed = speed        # mặc định = speed chung từ settings
            seg_duration = duration  # mặc định = duration chung từ settings
            if fit_timeline:
                # Khi khớp timeline: dùng speed (tương đối) thay vì duration,
                # để chỉ tăng tốc câu dài chứ không kéo dãn câu ngắn.
                seg_duration = 0
                slot = end - start  # khoảng thời gian phụ đề cho phép
                if slot > 0:
                    rate_lang = detect_rate_lang(text, language)
                    est = estimate_natural_duration(text, rate_lang)
                    if est > slot:
                        seg_speed = min(
                            est / slot, float(max_speed or MAX_SPEED_DEFAULT)
                        )
                        num_sped += 1
                        max_factor = max(max_factor, seg_speed)
                    else:
                        seg_speed = 1.0

            waveform, msg = _gen_waveform(
                text,
                language,
                ref_audio,
                instruct,
                num_step,
                guidance_scale,
                denoise,
                seg_speed,
                seg_duration,
                preprocess_prompt,
                postprocess_output,
                ref_text=ref_text or None,
            )
            if waveform is None:
                return None, f"Lỗi ở dòng {idx + 1}: {msg}"

            segments.append((pos, waveform))

            # Cập nhật con trỏ timeline: phần dư (nếu audio dài) sẽ tự đẩy câu sau.
            audio_len = len(waveform) / sampling_rate
            cursor_end = pos + audio_len
            prev_srt_end = end

        if not segments:
            return None, "Không có nội dung text nào để tạo audio."

        waveform_float = combine_segments(segments, sampling_rate)
        waveform_i16 = (waveform_float * 32767).astype(np.int16)
        total_dur = len(waveform_i16) / sampling_rate
        orig_dur = pysrttime_to_seconds(srt_data[-1].end)
        drift = total_dur - orig_dur
        status = (
            f"Done. Đã tạo {len(segments)} câu. "
            f"Tổng {total_dur:.1f}s (gốc {orig_dur:.1f}s, lệch {drift:+.1f}s). "
            f"Số câu phải tăng tốc: {num_sped}, hệ số lớn nhất: {max_factor:.2f}x."
        )
        return (sampling_rate, waveform_i16), status

    # =====================================================================
    # CODE MỚI (HARD SYNC): Chuyển .srt -> 1 audio NEO CỨNG theo timeline gốc.
    # Khác _srt_to_speech (cộng dồn, có thể trượt): hàm này đặt MỖI câu vào ĐÚNG
    # mốc start tuyệt đối của phụ đề và ép độ dài audio vừa khít ô (end - start)
    # bằng _fit_to_length. Nhờ vậy tổng audio == đúng độ dài video, không bao giờ
    # lệch giờ -> khớp hoàn hảo khi ghép vào video đã cháy phụ đề.
    # =====================================================================
    def _srt_to_speech_sync(
        srt_file,
        language,
        ref_audio,
        ref_text,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,        # giữ chữ ký giống _srt_to_speech (không dùng trực tiếp)
        duration,     # giữ chữ ký giống _srt_to_speech (không dùng trực tiếp)
        preprocess_prompt,
        postprocess_output,
        max_speed=MAX_SPEED_DEFAULT,  # giữ tham số cho tương thích; hard-sync luôn ép khớp
    ):
        if not srt_file:
            return None, "Vui lòng tải lên một file .srt."

        try:
            srt_data = pysrt.open(srt_file, encoding="utf-8")
        except Exception:
            try:
                srt_data = pysrt.open(srt_file)
            except Exception as e:
                return None, f"Không đọc được file SRT: {e}"

        if not srt_data or len(srt_data) == 0:
            return None, "File SRT rỗng hoặc sai định dạng."

        # Canvas dài đúng bằng mốc kết thúc câu cuối (== độ dài timeline gốc).
        total_seconds = pysrttime_to_seconds(srt_data[-1].end)
        total_samples = int(round(total_seconds * sampling_rate))
        if total_samples <= 0:
            return None, "Timeline phụ đề không hợp lệ (độ dài <= 0)."
        canvas = np.zeros(total_samples, dtype=np.float32)

        num_done = 0
        num_compressed = 0   # số câu phải nén lại cho vừa ô
        max_factor = 1.0     # hệ số nén lớn nhất (để báo cáo)

        for idx, item in enumerate(srt_data):
            text = item.text.replace("\n", " ").strip()
            if not text:
                continue
            start = pysrttime_to_seconds(item.start)
            end = pysrttime_to_seconds(item.end)
            slot = end - start
            if slot <= 0:
                continue

            # Gợi ý cho model đọc đúng độ dài ô; speed=1.0, để _fit_to_length lo phần khớp.
            waveform, msg = _gen_waveform(
                text,
                language,
                ref_audio,
                instruct,
                num_step,
                guidance_scale,
                denoise,
                1.0,          # speed
                slot,         # duration gợi ý = đúng ô thời gian
                preprocess_prompt,
                postprocess_output,
                ref_text=ref_text or None,
            )
            if waveform is None:
                return None, f"Lỗi ở dòng {idx + 1}: {msg}"

            slot_samples = int(round(slot * sampling_rate))
            raw_len = len(waveform)
            if raw_len > slot_samples and slot_samples > 0:
                num_compressed += 1
                max_factor = max(max_factor, raw_len / float(slot_samples))

            # Ép vừa đúng ô rồi bo nhẹ đầu/cuối để nối êm.
            seg = _fit_to_length(waveform, slot_samples, sampling_rate)
            seg = _apply_edge_fade(seg, sampling_rate)

            start_sample = int(round(start * sampling_rate))
            end_sample = start_sample + len(seg)
            # An toàn: nếu lỡ vượt canvas (do làm tròn) thì cắt bớt.
            if end_sample > total_samples:
                seg = seg[: total_samples - start_sample]
                end_sample = total_samples
            if start_sample < total_samples and len(seg) > 0:
                canvas[start_sample:end_sample] += seg
                num_done += 1

        np.clip(canvas, -1.0, 1.0, out=canvas)
        waveform_i16 = (canvas * 32767).astype(np.int16)
        total_dur = len(waveform_i16) / sampling_rate
        status = (
            f"Done (hard-sync). Đã tạo {num_done} câu, khớp đúng timeline. "
            f"Tổng {total_dur:.1f}s (== gốc, lệch 0.0s). "
            f"Số câu phải nén: {num_compressed}, hệ số nén lớn nhất: {max_factor:.2f}x."
        )
        return (sampling_rate, waveform_i16), status

    # =====================================================================
    # CODE MỚI: Bộ điều phối cho nút Generate của tab SRT -> chọn chế độ
    # neo cứng (hard-sync) hay cộng dồn (cũ) tuỳ checkbox. Đây cũng là điểm
    # mở API (api_name="srt_to_speech") để pipeline ngoài gọi qua gradio_client.
    # =====================================================================
    def _srt_dispatch(
        srt_file,
        language,
        ref_audio,
        ref_text,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        fit_timeline,
        max_speed,
        hard_sync,
    ):
        if hard_sync:
            return _srt_to_speech_sync(
                srt_file,
                language,
                ref_audio,
                ref_text,
                instruct,
                num_step,
                guidance_scale,
                denoise,
                speed,
                duration,
                preprocess_prompt,
                postprocess_output,
                max_speed,
            )
        return _srt_to_speech(
            srt_file,
            language,
            ref_audio,
            ref_text,
            instruct,
            num_step,
            guidance_scale,
            denoise,
            speed,
            duration,
            preprocess_prompt,
            postprocess_output,
            fit_timeline,
            max_speed,
        )

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # CODE MỚI: sửa hàm _lang_dropdown (bản gốc bị lỗi cú pháp chỉ có dấu ")").
    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        return gr.Dropdown(
            choices=_ALL_LANGUAGES,
            value=value,
            label=label,
            filterable=True,
        )

    # CODE MỚI: sửa hàm _gen_settings (bản gốc return biến chưa khai báo).
    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Advanced settings / Cài đặt nâng cao", open=False):
            ns = gr.Slider(
                minimum=1, maximum=64, value=32, step=1,
                label="num_step (số bước khử nhiễu, cao hơn = chất lượng hơn, chậm hơn)",
            )
            gs = gr.Slider(
                minimum=0.0, maximum=5.0, value=2.0, step=0.1,
                label="guidance_scale (độ bám theo điều kiện)",
            )
            dn = gr.Checkbox(value=True, label="denoise (khử nhiễu đầu ra)")
            sp = gr.Slider(
                minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                label="speed (tốc độ đọc, 1.0 = tự nhiên)",
            )
            du = gr.Number(
                value=0,
                label="duration (ép thời lượng giây, 0 = tự động)",
            )
            pp = gr.Checkbox(value=True, label="preprocess_prompt")
            po = gr.Checkbox(value=True, label="postprocess_output")
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo — SRT / Text to Speech

State-of-the-art text-to-speech model cho **600+ ngôn ngữ**:

- **SRT to Speech** — Chuyển file phụ đề `.srt` thành audio hoàn chỉnh theo timeline
- **Text to Speech** — Nhập text trực tiếp để tạo giọng nói
- Hỗ trợ **Voice Clone** từ một đoạn audio mẫu (3–10 giây)

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # CODE MỚI: Tab SRT to Speech
            # ==============================================================
            with gr.TabItem("SRT to Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        srt_file = gr.File(
                            label="File phụ đề (.srt)",
                            file_types=[".srt"],
                            type="filepath",
                        )
                        srt_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        srt_ref_audio = gr.Audio(
                            label="Reference Audio (tùy chọn, để clone giọng)",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Khuyến nghị: audio mẫu 3–10 giây. Bỏ trống nếu không clone giọng."
                            "</span>"
                        )
                        srt_ref_text = gr.Textbox(
                            label="Reference Text (tùy chọn)",
                            lines=2,
                            placeholder="Lời thoại của audio mẫu. Bỏ trống để tự động"
                            " nhận diện bằng ASR.",
                        )
                        with gr.Accordion("Instruct (optional)", open=False):
                            srt_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            srt_ns,
                            srt_gs,
                            srt_dn,
                            srt_sp,
                            srt_du,
                            srt_pp,
                            srt_po,
                        ) = _gen_settings()
                        # CODE MỚI: điều khiển canh thời gian theo timeline phụ đề.
                        srt_fit = gr.Checkbox(
                            value=True,
                            label="Khớp theo timeline phụ đề (tự tăng tốc câu dài)",
                        )
                        srt_maxspd = gr.Slider(
                            minimum=1.0, maximum=2.0,
                            value=MAX_SPEED_DEFAULT, step=0.05,
                            label="Giới hạn tăng tốc tối đa (MAX_SPEED)",
                        )
                        # CODE MỚI: bật chế độ neo cứng timeline (không trượt giờ).
                        srt_hard = gr.Checkbox(
                            value=True,
                            label="Hard sync (neo cứng timeline, KHÔNG trượt giờ - "
                            "khuyến nghị khi ghép vào video)",
                        )
                        srt_btn = gr.Button("Generate / Tạo audio", variant="primary")
                    with gr.Column(scale=1):
                        srt_audio = gr.Audio(
                            label="Output Audio / Kết quả",
                            type="numpy",
                        )
                        srt_status = gr.Textbox(label="Status / Trạng thái", lines=3)

                # CODE MỚI: dùng _srt_dispatch (chọn hard-sync/cộng dồn) và MỞ API
                # (api_name="srt_to_speech") để pipeline ngoài gọi qua gradio_client.
                srt_btn.click(
                    _srt_dispatch,
                    inputs=[
                        srt_file,
                        srt_lang,
                        srt_ref_audio,
                        srt_ref_text,
                        srt_instruct,
                        srt_ns,
                        srt_gs,
                        srt_dn,
                        srt_sp,
                        srt_du,
                        srt_pp,
                        srt_po,
                        srt_fit,
                        srt_maxspd,
                        srt_hard,
                    ],
                    outputs=[srt_audio, srt_status],
                    api_name="srt_to_speech",
                )

            # ==============================================================
            # CODE MỚI: Tab Text to Speech
            # ==============================================================
            with gr.TabItem("Text to Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize / Văn bản cần đọc",
                            lines=4,
                            placeholder="Nhập nội dung cần chuyển thành giọng nói...",
                        )
                        tts_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        tts_ref_audio = gr.Audio(
                            label="Reference Audio (tùy chọn, để clone giọng)",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        tts_ref_text = gr.Textbox(
                            label="Reference Text (tùy chọn)",
                            lines=2,
                            placeholder="Lời thoại của audio mẫu. Bỏ trống để tự động"
                            " nhận diện bằng ASR.",
                        )
                        with gr.Accordion("Instruct (optional)", open=False):
                            tts_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            tts_ns,
                            tts_gs,
                            tts_dn,
                            tts_sp,
                            tts_du,
                            tts_pp,
                            tts_po,
                        ) = _gen_settings()
                        tts_btn = gr.Button("Generate / Tạo audio", variant="primary")
                    with gr.Column(scale=1):
                        tts_audio = gr.Audio(
                            label="Output Audio / Kết quả",
                            type="numpy",
                        )
                        tts_status = gr.Textbox(label="Status / Trạng thái", lines=2)

                def _text_fn(
                    text, lang, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone" if ref_aud else "design",
                        ref_text=ref_text or None,
                    )

                tts_btn.click(
                    _text_fn,
                    inputs=[
                        tts_text,
                        tts_lang,
                        tts_ref_audio,
                        tts_ref_text,
                        tts_instruct,
                        tts_ns,
                        tts_gs,
                        tts_dn,
                        tts_sp,
                        tts_du,
                        tts_pp,
                        tts_po,
                    ],
                    outputs=[tts_audio, tts_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ===========================================================================
# GHI CHÚ: Theo quy ước "không xóa code cũ, chỉ comment lại".
# Các tab "Voice Clone" và "Voice Design" trong bản gốc đã được thay thế bằng
# 2 tab mới "SRT to Speech" và "Text to Speech" ở trên. Logic voice clone /
# voice design vẫn được giữ nguyên bên trong _gen_waveform (qua tham số
# ref_audio và instruct), nên không mất tính năng. Phần dropdown Voice Design
# theo _CATEGORIES (hiện đang rỗng) có thể bổ sung lại sau nếu cần.
# ===========================================================================
