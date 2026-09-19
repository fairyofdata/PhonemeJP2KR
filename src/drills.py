"""Targeted drill sentences, one set per detectable L1 error tag.

Stages follow the sequence proposed for Japanese learners by Lee (2022,
see docs/L1_TAXONOMY.md): monophthongs → diphthongs → onset laryngeal
contrast → syllable structure → codas. Only categories the jamo-level
scorer can actually detect get a drill; phonological-rule application,
voicing and intonation are out of reach of a text-output ASR (see the
coverage audit in docs/L1_TAXONOMY.md), so drilling them here would give
feedback the system cannot back with evidence.

Each sentence concentrates the target contrast so that the error tag the
drill trains is also the one most likely to fire when it is missed.
"""

DRILLS = [
    {
        "tag": "vowel_ʌ_o_confusion",
        "stage": "1. 単母音",
        "label": "ㅓ と ㅗ の区別",
        "sentences": ["서울에 가요", "거기 고기 있어요", "어머니가 오셨어요"],
    },
    {
        "tag": "vowel_ɯ_u_confusion",
        "stage": "1. 単母音",
        "label": "ㅡ と ㅜ の区別",
        "sentences": ["그 구두 주세요", "음악을 들어요", "은행은 어디예요"],
    },
    {
        "tag": "vowel_jʌ_jo_confusion",
        "stage": "2. 二重母音",
        "label": "ㅕ と ㅛ の区別",
        "sentences": ["여기요", "여름에 여행 가요", "요즘 영어를 배워요"],
    },
    {
        "tag": "diphthong_ɰi_monophthongization",
        "stage": "2. 二重母音",
        "label": "語頭の ㅢ",
        "sentences": ["의사 선생님", "의자에 앉으세요", "의미가 뭐예요"],
    },
    {
        "tag": "laryngeal_confusion",
        "stage": "3. 初声子音",
        "label": "平音・激音・濃音",
        "sentences": ["방에 빵이 있어요", "달하고 딸", "비싼 피자를 샀어요"],
    },
    {
        "tag": "vowel_epenthesis",
        "stage": "4. 音節構造",
        "label": "パッチムの後に母音を入れない",
        "sentences": ["국밥 한 그릇 주세요", "학교 앞 꽃집", "책 한 권"],
    },
    {
        "tag": "coda_deletion",
        "stage": "4. 音節構造",
        "label": "パッチムを落とさない",
        "sentences": ["밥 먹었어요", "옷 샀어요", "물 한 잔"],
    },
    {
        "tag": "nasal_coda_confusion",
        "stage": "5. 終声",
        "label": "ㄴ・ㅁ・ㅇ パッチム",
        "sentences": ["산에서 상을 받았어요", "감기 조심하세요", "방 안이 밝아요"],
    },
    {
        "tag": "stop_coda_confusion",
        "stage": "5. 終声",
        "label": "ㄱ・ㄷ・ㅂ パッチム",
        "sentences": ["밥 박 밭", "학교 입구에서 만나요", "숟가락 주세요"],
    },
]

DRILL_BY_TAG = {d["tag"]: d for d in DRILLS}
