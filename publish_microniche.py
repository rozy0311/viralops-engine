"""
ViralOps Engine — Micro-Niche Plant-Based Content Publisher
Generates CORRECT content following the Universal Caption + Hashtag Matrix strategy.

Content Pack format:
  1) Title — keyword-rich, descriptive
  2) Content — ~4000 chars, witty conversational, educational
  3) Universal Caption — [LOCATION] [SEASON] [PAIN POINT] | [AUDIENCE] | Steps + CTA + Hashtags
  4) 9:16 Image — clean, minimal text overlay
"""
import sys, os, json, random, tempfile, time

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()

from PIL import Image, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════
# CONTENT STRATEGY DATABASE — From spec files
# ═══════════════════════════════════════════════════════════════

# 20 Micro-Niches (plant-based raw materials for TikTok US)
MICRO_NICHES = [
    "almonds", "chickpeas", "quinoa", "chia seeds", "flaxseeds",
    "hemp seeds", "oats", "pepitas", "sunflower seeds", "lentils",
    "sesame seeds", "walnuts", "black beans", "pecans", "psyllium husk",
    "cashews", "coconut", "spirulina", "cacao nibs", "pumpkin seeds",
]

# Nano-niches (ultra-specific personas)
NANO_NICHES = [
    "pregnancy keto almonds", "AIP chickpeas autoimmune", "trucker quinoa meal prep",
    "RV camping chia pudding", "bulletproof coffee flaxseed", "postpartum oats recovery",
    "cashew soak histamine intolerance", "spirulina mask acne-prone skin",
    "psyllium husk gut reset", "sunflower butter nut-free kids",
]

# Real-life everyday content topics
REAL_LIFE_NICHES = [
    "banana peel water houseplants apartment", "green onion regrow water office worker",
    "freeze spinach smoothie bags gym commuter", "strawberry tops vinegar salad dressing",
    "herbs regrow from scraps kitchen window", "holy basil tulsi oral health",
    "DIY fabric refresher spray natural", "indoor grow lights apartment vegetables",
    "harvest-to-table same day routine", "beginner mistakes killing vegetables",
]

# Locations to rotate for Universal Caption
LOCATIONS = [
    "Chicago", "NYC", "LA", "Houston", "Phoenix", "Denver",
    "Portland", "Seattle", "Austin", "Miami", "Atlanta", "Boston",
]

# Seasons
SEASONS = ["Winter", "Spring", "Summer", "Fall"]

# Hashtag layers by niche
HASHTAG_MATRIX = {
    "almonds": {
        "broad": ["#plantbased", "#veganrecipes", "#healthysnacks"],
        "micro": ["#rawalmonds", "#almondmilk", "#almondbutter"],
        "nano": ["#soakalmonds", "#almondbenefits", "#nutmilkathome"],
        "trend": ["#wellnesstok", "#healthytiktok"],
    },
    "chickpeas": {
        "broad": ["#plantbased", "#veganprotein", "#mealprep"],
        "micro": ["#chickpeas", "#hummus", "#roastedchickpeas"],
        "nano": ["#chickpeasprout", "#aquafaba", "#cheapprotein"],
        "trend": ["#budgetvegan", "#proteinhack"],
    },
    "quinoa": {
        "broad": ["#plantbased", "#healthyeating", "#mealprep"],
        "micro": ["#quinoa", "#quinoabowl", "#quinoarecipes"],
        "nano": ["#quinoaprotein", "#glutenfreemeal", "#grainbowl"],
        "trend": ["#lunchideas", "#mealprepmonday"],
    },
    "chia seeds": {
        "broad": ["#plantbased", "#healthybreakfast", "#nutrition"],
        "micro": ["#chiaseeds", "#chiapudding", "#overnightchia"],
        "nano": ["#chiarecipes", "#fiberfood", "#omegaseeds"],
        "trend": ["#breakfastideas", "#cleaneating"],
    },
    "spirulina": {
        "broad": ["#plantbased", "#superfood", "#wellness"],
        "micro": ["#spirulina", "#spirulinapowder", "#greenpower"],
        "nano": ["#spirulinamask", "#algaefood", "#acnespirulina"],
        "trend": ["#skincaretok", "#wellnesstok"],
    },
    "sunflower seeds": {
        "broad": ["#plantbased", "#veganrecipes", "#healthysnacks"],
        "micro": ["#sunflowerbutter", "#sunflowerseeds", "#nutfree"],
        "nano": ["#seedbutter", "#allergyfreefood", "#budgetvegan"],
        "trend": ["#kitchenhacks", "#snackideas"],
    },
    "psyllium husk": {
        "broad": ["#fiber", "#guthealth", "#digestion"],
        "micro": ["#psylliumhusk", "#rawfiber", "#gutbalance"],
        "nano": ["#constipationrelief", "#fibersupplement", "#bloating"],
        "trend": ["#wellness", "#nutrition"],
    },
    "cashews": {
        "broad": ["#plantbased", "#veganrecipes", "#dairyfree"],
        "micro": ["#cashewcream", "#cashewmilk", "#rawcashews"],
        "nano": ["#cashewsoak", "#histamine", "#cashewcheese"],
        "trend": ["#veganfood", "#dairyfreelife"],
    },
    "default": {
        "broad": ["#plantbased", "#vegan", "#healthyeating"],
        "micro": ["#plantbasedrecipes", "#wholefood", "#eatclean"],
        "nano": ["#plantpower", "#veganlife", "#naturalfood"],
        "trend": ["#healthytiktok", "#wellnesstok"],
    },
}

# Pre-written content packs (from spec files) — can publish directly
PRE_WRITTEN_PACKS = [
    {
        "title": "Regrow Green Onions in Water: The Desk-to-Fridge Hack for Office Workers",
        "pain_point": "Always out of green onions for meal prep?",
        "audiences": ["Office workers", "Apartment renters", "Beginners"],
        "steps": [
            "Save the root ends (1-2 inches)",
            "Jar + water (roots only) + fridge shelf",
        ],
        "result": "Fresh onions in 1-3 days (trim + repeat)",
        "hashtags": ["#mealprep", "#officehacks", "#houseplants", "#zerowaste", "#kitchenhacks"],
        "image_title": "Regrow Green Onions in Water",
        "image_subtitle": "Desk-to-Fridge Hack",
        "image_steps": "Save • Soak • Trim • Repeat",
        "colors": ((20, 80, 50), (40, 150, 90)),
    },
    {
        "title": "Freeze Spinach Smoothie Bags: Grab-Blend-Go for Gym Commuters",
        "pain_point": "Spinach keeps going slimy before you can use it?",
        "audiences": ["Gym commuters", "Busy workers", "Meal prep beginners"],
        "steps": [
            "Portion spinach + banana + fruit into zip bags",
            "Freeze flat + stack",
        ],
        "result": "Grab-blend-go smoothies all week",
        "hashtags": ["#FreezeSpinach", "#SmoothieBags", "#GymCommute", "#DailyGreen", "#Prepped"],
        "image_title": "Freeze Spinach Smoothie Bags",
        "image_subtitle": "Meal Prep in 10 Min",
        "image_steps": "Grab • Blend • Go",
        "colors": ((20, 60, 30), (50, 140, 70)),
    },
    {
        "title": "Strawberry Tops Vinegar Salad Dressing: Zero-Waste Spring Vinaigrette",
        "pain_point": "Throwing strawberry tops away?",
        "audiences": ["Busy cooks", "Salad beginners", "Zero-waste people"],
        "steps": [
            "Soak strawberry tops in vinegar (12-48h)",
            "Strain + shake with olive oil",
        ],
        "result": "Fresh cafe-style vinaigrette",
        "hashtags": ["#zerowaste", "#saladdressing", "#vinaigrette", "#healthyrecipes", "#foodwaste"],
        "image_title": "Strawberry Tops Vinegar",
        "image_subtitle": "Zero-Waste Vinaigrette",
        "image_steps": "Soak • Strain • Shake",
        "colors": ((120, 30, 40), (200, 70, 80)),
    },
    {
        "title": "Banana Peel Water for Houseplants: Apartment-Friendly Plant Fertilizer Hack",
        "pain_point": "Houseplants looking sad in your apartment?",
        "audiences": ["Apartment renters", "Plant beginners", "Budget people"],
        "steps": [
            "Soak banana peel in water (12-24h)",
            "Strain + water like normal",
        ],
        "result": "Easy plant boost (zero-waste + apartment-friendly)",
        "hashtags": ["#houseplants", "#plantcare", "#indoorplants", "#urbangarden", "#apartmentliving"],
        "image_title": "Banana Peel Water",
        "image_subtitle": "Apartment Plant Hack",
        "image_steps": "Soak • Strain • Water",
        "colors": ((100, 80, 10), (180, 150, 40)),
    },
    {
        "title": "Raw Sunflower Seeds for Butter: Nut-Free, Budget-Friendly Plant Spread",
        "pain_point": "Nut butter too expensive (or allergies)?",
        "audiences": ["Busy people", "Budget vegans", "Nut-free families"],
        "steps": [
            "Blend raw sunflower seeds",
            "Add pinch of salt (optional vanilla/cinnamon)",
        ],
        "result": "Creamy nut-free spread (cheaper + fresher)",
        "hashtags": ["#plantbased", "#veganrecipes", "#healthysnacks", "#sunflowerbutter", "#nutfree"],
        "image_title": "Sunflower Seed Butter",
        "image_subtitle": "Nut-Free & Budget",
        "image_steps": "Blend • Season • Spread",
        "colors": ((100, 70, 20), (190, 140, 50)),
    },
    {
        "title": "Spirulina Mask for Acne-Prone Vegan Skin: 2-Min Mix",
        "pain_point": "Acne acting up but harsh masks wreck your skin?",
        "audiences": ["Vegan skincare", "Acne-prone people", "Beginners"],
        "steps": [
            "Mix spirulina + aloe gel",
            "Add oat milk (optional kaolin clay)",
        ],
        "result": "Calm glow in 5-8 minutes",
        "hashtags": ["#acnespirulina", "#rawmaskvegan", "#plantacne", "#teenvegan", "#glow"],
        "image_title": "Spirulina Mask",
        "image_subtitle": "Acne-Prone Skin Fix",
        "image_steps": "Mix • Apply • Rinse",
        "colors": ((10, 60, 50), (30, 140, 110)),
    },
    {
        "title": "Holy Basil (Tulsi) for Oral Health: Natural Remedy for Gums & Fresh Breath",
        "pain_point": "Bad breath + gum irritation ruining your confidence?",
        "audiences": ["Natural wellness people", "Sensitive gums", "Beginners"],
        "steps": [
            "Steep holy basil (Tulsi) in warm water",
            "Swish 30 seconds (after brushing)",
        ],
        "result": "Fresher breath + calmer gums",
        "hashtags": ["#oralhealth", "#dentalcare", "#naturalremedy", "#ayurveda", "#badbreath"],
        "image_title": "Holy Basil Tulsi",
        "image_subtitle": "Oral Health Routine",
        "image_steps": "Steep • Swish • Rinse",
        "colors": ((30, 70, 30), (60, 140, 60)),
    },
    {
        "title": "Psyllium Husk Raw Fiber: The Mix-Drink Gut Reset That Actually Works",
        "pain_point": "Digestive issues ruining your routine?",
        "audiences": ["Busy workers", "Gut-health beginners", "Wellness people"],
        "steps": [
            "Mix 1 tsp psyllium with a full glass of water",
            "Drink fast + hydrate after",
        ],
        "result": "More regular digestion in 1-3 days",
        "hashtags": ["#fiber", "#guthealth", "#digestion", "#nutrition", "#wellness"],
        "image_title": "Psyllium Husk Fiber",
        "image_subtitle": "Gut Reset in Minutes",
        "image_steps": "Mix • Drink • Hydrate",
        "colors": ((60, 40, 20), (140, 100, 50)),
    },
]


# ═══════════════════════════════════════════════════════════════
# Gemini prompt template for generating NEW micro-niche content
# ═══════════════════════════════════════════════════════════════

def build_gemini_prompt(niche: str, location: str, season: str) -> str:
    """Build a Gemini prompt that follows the exact content pack format."""
    return f"""You are the content creator for "The Rike Root Stories" TikTok account.
This is a PLANT-BASED micro-niche account focusing on raw materials, natural health, 
zero-waste kitchen hacks, and budget-friendly vegan living for US audiences.

Create ONE complete content pack for the micro-niche topic: "{niche}"

The content MUST follow this EXACT structure. Return as JSON:

{{
  "title": "Keyword-rich descriptive title (50-80 chars)",
  "pain_point": "A relatable pain point question for [{location}] [{season}] (max 60 chars)",
  "audiences": ["Audience 1", "Audience 2", "Audience 3"],
  "step1": "Clear step 1 with specific action (max 50 chars)",
  "step2": "Clear step 2 with specific action (max 50 chars)",
  "result": "The outcome with benefit (max 50 chars)",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"],
  "image_title": "Short title for image overlay (max 30 chars)",
  "image_subtitle": "Short subtitle (max 25 chars)",
  "image_steps": "3 words joined by bullet dots like: Word1 • Word2 • Word3"
}}

RULES:
- Content must be about PLANT-BASED raw materials, natural health, or zero-waste kitchen
- Pain point should use [{location}] [{season}] format at start
- Hashtags must be high-search, relevant to the niche
- Tone: witty, conversational, educational — NOT salesy
- Target US audience, especially budget-conscious, apartment renters, beginners
- NO generic self-improvement or morning routines
- Return ONLY valid JSON, no markdown fences"""


# ═══════════════════════════════════════════════════════════════
# IMAGE GENERATION — 9:16 Clean Minimal Aesthetic
# ═══════════════════════════════════════════════════════════════

W, H = 1080, 1920

# Color palettes for different niche categories
COLOR_PALETTES = [
    ((20, 80, 50), (40, 160, 90)),       # Forest green (plants/greens)
    ((30, 50, 90), (60, 100, 170)),       # Deep blue (wellness)
    ((80, 50, 20), (170, 110, 50)),       # Warm amber (food/seeds)
    ((60, 30, 60), (130, 70, 130)),       # Purple (health/beauty)
    ((20, 70, 70), (50, 150, 140)),       # Teal (fresh/clean)
    ((90, 40, 30), (190, 80, 60)),        # Terracotta (natural)
    ((40, 60, 30), (90, 130, 60)),        # Olive green (organic)
    ((70, 50, 30), (150, 110, 60)),       # Earth tone (raw materials)
]


def make_gradient(width, height, color1, color2):
    """Create smooth vertical gradient."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        ratio = y / height
        r = int(color1[0] + (color2[0] - color1[0]) * ratio)
        g = int(color1[1] + (color2[1] - color1[1]) * ratio)
        b = int(color1[2] + (color2[2] - color1[2]) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img


def wrap_text(text, font, max_width, draw):
    """Word-wrap text to fit within max_width."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def load_fonts():
    """Load fonts with fallback."""
    for font_path in [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]:
        if os.path.exists(font_path):
            return {
                "title": ImageFont.truetype(font_path, 80),
                "subtitle": ImageFont.truetype(font_path, 56),
                "body": ImageFont.truetype(font_path, 48),
                "small": ImageFont.truetype(font_path, 40),
                "brand": ImageFont.truetype(font_path, 36),
            }
    default = ImageFont.load_default()
    return {"title": default, "subtitle": default, "body": default, "small": default, "brand": default}


def generate_post_image(pack: dict, tmpdir: str) -> str:
    """Generate a single 9:16 TikTok image for the content pack."""
    fonts = load_fonts()
    colors = pack.get("colors", random.choice(COLOR_PALETTES))

    img = make_gradient(W, H, colors[0], colors[1])
    draw = ImageDraw.Draw(img)

    # ── Brand watermark at top ──
    draw.text((W // 2, 160), "THE RIKE ROOT STORIES", fill=(255, 255, 255, 180),
              font=fonts["brand"], anchor="mm")

    # ── Decorative line ──
    line_y = 220
    draw.line([(W // 2 - 200, line_y), (W // 2 + 200, line_y)],
              fill=(255, 255, 255, 100), width=2)

    # ── Main title ──
    title_text = pack.get("image_title", pack["title"][:30])
    title_lines = wrap_text(title_text.upper(), fonts["title"], W - 120, draw)
    y = 400
    for line in title_lines:
        draw.text((W // 2, y), line, fill=(255, 255, 255), font=fonts["title"], anchor="mm")
        y += 100

    # ── Subtitle ──
    subtitle = pack.get("image_subtitle", "")
    if subtitle:
        draw.text((W // 2, y + 60), subtitle, fill=(220, 220, 200), font=fonts["subtitle"], anchor="mm")

    # ── Steps in center ──
    steps_text = pack.get("image_steps", "Learn • Try • Share")
    steps_y = H // 2 + 100
    draw.text((W // 2, steps_y), steps_text, fill=(255, 255, 255), font=fonts["body"], anchor="mm")

    # ── Pain point question at bottom ──
    pain = pack.get("pain_point", "")
    if pain:
        pain_lines = wrap_text(pain, fonts["small"], W - 100, draw)
        py = H - 500
        for line in pain_lines:
            draw.text((W // 2, py), line, fill=(200, 200, 180), font=fonts["small"], anchor="mm")
            py += 55

    # ── CTA ──
    draw.text((W // 2, H - 300), "Full tutorial on profile!", fill=(255, 255, 200),
              font=fonts["small"], anchor="mm")

    # ── Bottom brand ──
    draw.text((W // 2, H - 120), "@TheRikeRootStories", fill=(180, 180, 180),
              font=fonts["brand"], anchor="mm")

    path = os.path.join(tmpdir, "microniche_post.jpg")
    img.save(path, "JPEG", quality=95)
    return path


# ═══════════════════════════════════════════════════════════════
# BUILD UNIVERSAL CAPTION from content pack
# ═══════════════════════════════════════════════════════════════

def build_caption(pack: dict, location: str, season: str) -> str:
    """Build the Universal Caption following the spec template."""
    pain = pack.get("pain_point", "Looking for a healthier routine?")
    audiences = pack.get("audiences", ["Beginners", "Busy people", "Budget vegans"])
    steps = pack.get("steps", [pack.get("step1", "Step 1"), pack.get("step2", "Step 2")])
    result_text = pack.get("result", "Easy plant-based win")
    hashtags = pack.get("hashtags", ["#plantbased", "#vegan", "#healthyeating", "#wellness", "#tiktok"])

    # Universal Caption format from spec
    caption_parts = [
        f"[{location}] [{season}] {pain} \u2744\ufe0f",
        " | ".join(f"{a}?" for a in audiences),
        "",
    ]

    # Steps
    for i, step in enumerate(steps, 1):
        caption_parts.append(f"\u2022 Step {i}: {step}")

    caption_parts.append(f"\u2022 Result: {result_text} \u2728")
    caption_parts.append("")
    caption_parts.append("Full tutorial pinned on my profile! \U0001f447")
    caption_parts.append("")
    caption_parts.append(" ".join(hashtags))

    return "\n".join(caption_parts)


# ═══════════════════════════════════════════════════════════════
# MODE SELECTION — Pre-written pack or Gemini-generated
# ═══════════════════════════════════════════════════════════════

def get_content_pack(mode: str = "auto") -> dict:
    """
    Get a content pack.
    mode='prewritten' → pick from pre-written packs
    mode='gemini' → generate new with Gemini
    mode='niche_hunter' → use Micro-Niche Hunter scored questions
    mode='pain_point' → use pain-point questions (urgent answers)
    mode='auto' → weighted random (40% prewritten, 30% gemini, 20% niche_hunter, 10% pain_point)
    """
    if mode == "auto":
        mode = random.choices(
            ["prewritten", "gemini", "niche_hunter", "pain_point"],
            weights=[40, 30, 20, 10],
            k=1,
        )[0]

    location = random.choice(LOCATIONS)
    season = random.choice(SEASONS)

    if mode == "prewritten":
        pack = random.choice(PRE_WRITTEN_PACKS)
        pack["_location"] = location
        pack["_season"] = season
        pack["_source"] = "prewritten"
        print(f"  Mode: Pre-written pack")
        print(f"  Title: {pack['title']}")
        return pack

    # ── Niche Hunter mode ──
    if mode in ("niche_hunter", "pain_point"):
        print(f"  Mode: Niche Hunter ({mode})")
        try:
            from niche_hunter import get_top_content_pack, get_pain_point_content, match_channel
            from niche_hunter import HIGHEST_SEARCH_QUESTIONS, PAIN_POINT_QUESTIONS, score_question, NicheScore

            if mode == "pain_point":
                pains = get_pain_point_content(top_n=5)
                if pains:
                    chosen = random.choice(pains)
                    topic = chosen["topic"]
                    niche_key = chosen.get("niche", "homesteading")
                else:
                    topic = "How to start homesteading in a small apartment?"
                    niche_key = "homesteading"
            else:
                # Pick top-scored question from random niche
                niche_keys = list(HIGHEST_SEARCH_QUESTIONS.keys())
                niche_key = random.choice(niche_keys)
                qs = HIGHEST_SEARCH_QUESTIONS[niche_key]
                scored = [score_question(q) for q in qs]
                scored.sort(key=lambda x: x.final_score, reverse=True)
                topic = scored[0].topic if scored else "How to grow mushrooms in apartment?"

            print(f"  Niche: {niche_key}")
            print(f"  Topic: {topic}")
            print(f"  Channel: @{match_channel(niche_key)}")
            print(f"  Score: {chosen['final_score'] if mode == 'pain_point' and pains else 'N/A'}")

            # Generate full content via Gemini using the high-scored topic
            from google import genai
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            prompt = build_gemini_prompt(topic, location, season)
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(text)
            pack = {
                "title": data["title"],
                "pain_point": data["pain_point"],
                "audiences": data["audiences"],
                "steps": [data["step1"], data["step2"]],
                "result": data["result"],
                "hashtags": data["hashtags"],
                "image_title": data.get("image_title", data["title"][:30]),
                "image_subtitle": data.get("image_subtitle", ""),
                "image_steps": data.get("image_steps", "Learn • Try • Share"),
                "colors": random.choice(COLOR_PALETTES),
                "_location": location,
                "_season": season,
                "_source": mode,
                "_niche": topic,
                "_niche_key": niche_key,
                "_channel": match_channel(niche_key),
            }
            print(f"  Title: {pack['title']}")
            return pack
        except ImportError:
            print("  ⚠️  niche_hunter.py not found — falling back to gemini mode")
        except Exception as e:
            print(f"  ⚠️  Niche Hunter error: {e} — falling back to gemini mode")

    # Gemini mode
    print(f"  Mode: Gemini-generated")
    niche = random.choice(MICRO_NICHES + NANO_NICHES + REAL_LIFE_NICHES)
    print(f"  Niche: {niche}")
    print(f"  Location: {location} | Season: {season}")

    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = build_gemini_prompt(niche, location, season)

    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = resp.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(text)

    # Build pack from Gemini response
    pack = {
        "title": data["title"],
        "pain_point": data["pain_point"],
        "audiences": data["audiences"],
        "steps": [data["step1"], data["step2"]],
        "result": data["result"],
        "hashtags": data["hashtags"],
        "image_title": data.get("image_title", data["title"][:30]),
        "image_subtitle": data.get("image_subtitle", ""),
        "image_steps": data.get("image_steps", "Learn • Try • Share"),
        "colors": random.choice(COLOR_PALETTES),
        "_location": location,
        "_season": season,
        "_source": "gemini",
        "_niche": niche,
    }

    # Add niche-specific hashtags if available
    niche_key = niche.split()[0].lower() if " " in niche else niche.lower()
    if niche_key in HASHTAG_MATRIX:
        matrix = HASHTAG_MATRIX[niche_key]
        extra_tags = matrix["micro"][:2] + matrix["nano"][:1] + matrix["trend"][:1]
        # Merge without duplicates
        existing = set(t.lower() for t in pack["hashtags"])
        for tag in extra_tags:
            if tag.lower() not in existing:
                pack["hashtags"].append(tag)
                existing.add(tag.lower())

    print(f"  Title: {pack['title']}")
    return pack


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    import httpx

    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    print("=" * 60)
    print(f"VIRALOPS MICRO-NICHE PUBLISHER (mode={mode})")
    print("=" * 60)

    # ── Step 1: Get content pack ──
    print("\n[1/5] Getting content pack...")
    pack = get_content_pack(mode)
    location = pack.get("_location", "Chicago")
    season = pack.get("_season", "Winter")

    # ── Step 2: Generate 9:16 image ──
    print("\n[2/5] Generating 9:16 TikTok image...")
    tmpdir = tempfile.mkdtemp(prefix="viralops_microniche_")
    image_path = generate_post_image(pack, tmpdir)
    print(f"  Image saved: {image_path}")

    # ── Step 3: Build Universal Caption ──
    print("\n[3/5] Building Universal Caption...")
    caption = build_caption(pack, location, season)
    print(f"  Caption ({len(caption)} chars):")
    print(f"  {caption[:300]}...")

    # ── Step 4: Upload to Publer + Publish ──
    print("\n[4/5] Uploading image & publishing to TikTok...")

    api_key = os.environ["PUBLER_API_KEY"]
    ws_id = os.environ.get("PUBLER_WORKSPACE_ID", "")
    tiktok_id = "698c95e5b1ab790def1352c1"

    auth_headers = {"Authorization": f"Bearer-API {api_key}"}
    if ws_id:
        auth_headers["Publer-Workspace-Id"] = ws_id

    # Upload image
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        r = httpx.post(
            "https://app.publer.com/api/v1/media",
            headers=auth_headers,
            files=files,
            timeout=30,
        )

    print(f"  Upload: {r.status_code}")
    if r.status_code not in (200, 201):
        print(f"  ERROR: {r.text[:500]}")
        sys.exit(1)

    media_data = r.json()
    media_entry = {"id": media_data.get("id", ""), "path": media_data.get("path", media_data.get("url", ""))}
    print(f"  Media: {json.dumps(media_entry)[:200]}")

    # Publish
    post_payload = {
        "bulk": {
            "state": "draft",
            "posts": [
                {
                    "networks": {
                        "tiktok": {
                            "type": "photo",
                            "text": caption,
                            "media": [media_entry],
                            "privacy_level": "PUBLIC_TO_EVERYONE",
                        }
                    },
                    "accounts": [{"id": tiktok_id}],
                }
            ],
        }
    }

    headers_json = {**auth_headers, "Content-Type": "application/json"}
    r = httpx.post(
        "https://app.publer.com/api/v1/posts/schedule/publish",
        headers=headers_json,
        json=post_payload,
        timeout=30,
    )

    print(f"  Publish: {r.status_code}")
    print(f"  Response: {r.text[:500]}")

    try:
        result = r.json()
    except Exception:
        result = {"raw": r.text}

    # ── Poll job status ──
    job_id = result.get("job_id", "") if isinstance(result, dict) else ""
    post_link = ""

    if job_id:
        print(f"\n  Job ID: {job_id}")
        print("  Polling...")
        for attempt in range(12):
            time.sleep(3)
            jr = httpx.get(
                f"https://app.publer.com/api/v1/job_status/{job_id}",
                headers=auth_headers,
                timeout=15,
            )
            if jr.status_code == 200:
                job_data = jr.json()
                status = job_data.get("status", "")
                print(f"    [{attempt + 1}] {status}")
                if status == "complete":
                    payload = job_data.get("payload", [])
                    if payload:
                        first = payload[0] if isinstance(payload, list) else payload
                        failure = first.get("failure", {})
                        post_info = first.get("post", {})
                        if failure:
                            print(f"\n  FAILED: {failure.get('message', str(failure))}")
                        elif post_info.get("state") in ("live", "published"):
                            post_link = post_info.get("post_link", "")
                            print(f"\n  SUCCESS! Post is LIVE!")
                            print(f"  Link: {post_link}")
                        else:
                            post_link = post_info.get("post_link", "")
                            print(f"\n  State: {post_info.get('state', 'unknown')}")
                            print(f"  Link: {post_link}")
                    break
            else:
                print(f"    [{attempt + 1}] HTTP {jr.status_code}")

    # ── Step 5: Save to DB ──
    print("\n[5/5] Saving to database...")
    try:
        from web.app import get_db_safe, init_db
        init_db()

        with get_db_safe() as conn:
            cur = conn.execute(
                "INSERT INTO posts (title, body, platforms, status, published_at, extra_fields) "
                "VALUES (?, ?, ?, ?, datetime('now'), ?)",
                (
                    pack["title"],
                    caption,
                    json.dumps(["tiktok"]),
                    "published",
                    json.dumps({
                        "hashtags": pack["hashtags"],
                        "source": f"microniche_{pack.get('_source', 'unknown')}",
                        "niche": pack.get("_niche", "prewritten"),
                        "location": location,
                        "season": season,
                        "publer_job_id": job_id,
                        "post_link": post_link,
                        "pain_point": pack.get("pain_point", ""),
                        "audiences": pack.get("audiences", []),
                    }),
                ),
            )
            post_id = cur.lastrowid
            conn.execute(
                "INSERT INTO publish_log (post_id, platform, success, post_url, error) VALUES (?, ?, ?, ?, ?)",
                (post_id, "tiktok", 1, post_link, ""),
            )
            conn.commit()
            print(f"  Saved as post #{post_id}")
    except Exception as e:
        print(f"  DB save failed (non-critical): {e}")

    print("\n" + "=" * 60)
    print("DONE! Micro-niche content published to TikTok.")
    print(f"Title: {pack['title']}")
    print(f"Source: {pack.get('_source', 'unknown')}")
    print(f"Niche: {pack.get('_niche', 'prewritten')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
