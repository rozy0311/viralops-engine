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

# ── Niche Hunter Top-Scored Packs (score 8.0+) ──
# Generated from niche_hunter.py 4-axis scoring system
NICHE_HUNTER_PACKS = [
    {
        "title": "How to Grow Mushrooms in a Dark Apartment — Zero Sunlight Needed",
        "pain_point": "No sunlight apartment — can you still grow food?",
        "audiences": ["Apartment dwellers", "Budget growers", "Beginners"],
        "steps": [
            "Get a mushroom grow kit (oyster or shiitake) — $15-20",
            "Place in closet or under sink — mist 2x/day, harvest in 10-14 days",
        ],
        "result": "Fresh mushrooms every 2 weeks from a dark closet",
        "hashtags": ["#mushroomgrowing", "#apartmentgarden", "#noSunlight", "#urbanfarming", "#growyourown"],
        "image_title": "Grow Mushrooms — No Sun",
        "image_subtitle": "Apartment Closet Farm",
        "image_steps": "Kit • Mist • Harvest",
        "colors": ((60, 40, 30), (140, 100, 70)),
        "_niche_score": 8.50,
    },
    {
        "title": "AI Content Flooding TikTok — 3 Free Tools to Detect Fake Posts",
        "pain_point": "Can't tell if what you're watching is AI-generated?",
        "audiences": ["Content creators", "Digital literacy", "Skeptical viewers"],
        "steps": [
            "Use Pangram (pangram.com) — paste text or upload image for AI detection",
            "Cross-check with GPTZero + Originality.ai free tiers",
        ],
        "result": "Spot AI-generated content in seconds — protect your feed",
        "hashtags": ["#AIdetection", "#fakenews", "#medialiteracy", "#TikTokTips", "#AIfact"],
        "image_title": "Detect AI Content",
        "image_subtitle": "3 Free Tools",
        "image_steps": "Paste • Scan • Verify",
        "colors": ((30, 40, 80), (60, 90, 180)),
        "_niche_score": 8.40,
    },
    {
        "title": "Grow Microgreens on Your Windowsill — Just a Tray and 5 Days",
        "pain_point": "Want to grow food but no garden, no space?",
        "audiences": ["Apartment renters", "Health-conscious", "Beginners"],
        "steps": [
            "Fill a shallow tray with 1 inch of soil, scatter seeds (broccoli/radish/sunflower)",
            "Mist daily, cover 2 days → uncover at window → harvest day 7-10",
        ],
        "result": "Nutrient-dense microgreens (40x more nutrients than mature plants)",
        "hashtags": ["#microgreens", "#windowgarden", "#growyourown", "#healthyfood", "#urbangarden"],
        "image_title": "Windowsill Microgreens",
        "image_subtitle": "5-Day Harvest",
        "image_steps": "Sow • Mist • Harvest",
        "colors": ((20, 70, 30), (50, 160, 70)),
        "_niche_score": 8.25,
    },
    {
        "title": "Grow Potatoes in a 5-Gallon Bucket — 25 lbs From Your Patio",
        "pain_point": "Food prices insane but zero garden space?",
        "audiences": ["Budget families", "Patio gardeners", "Survival skills"],
        "steps": [
            "Drill holes in bottom of 5-gallon bucket, fill 4 inches soil + 1 seed potato",
            "As sprouts grow, keep adding soil/straw to cover stems — harvest when plant dies back",
        ],
        "result": "Up to 25 lbs of potatoes from one bucket — $3 investment",
        "hashtags": ["#bucketgarden", "#growpotatoes", "#urbanfarming", "#foodsecurity", "#DIYgarden"],
        "image_title": "Potatoes in a Bucket",
        "image_subtitle": "25 lbs for $3",
        "image_steps": "Drill • Plant • Hill • Harvest",
        "colors": ((80, 60, 30), (170, 130, 60)),
        "_niche_score": 8.25,
    },
    {
        "title": "Homesteading in a 500sqft Apartment — The Starter Kit You Already Own",
        "pain_point": "Want to homestead but live in a tiny apartment?",
        "audiences": ["Urban beginners", "Small-space livers", "Budget homesteaders"],
        "steps": [
            "Start 3 things TODAY: windowsill herbs (basil, mint), countertop composting (bokashi), sprout jar",
            "Week 2: add mushroom kit under sink + fermentation jar (sauerkraut)",
        ],
        "result": "5 homesteading skills running in 500sqft — zero yard needed",
        "hashtags": ["#urbanhomesteading", "#apartmentgarden", "#smallspace", "#composting", "#sprouts"],
        "image_title": "500sqft Homestead",
        "image_subtitle": "Apartment Starter Kit",
        "image_steps": "Herbs • Compost • Sprout • Ferment",
        "colors": ((40, 70, 40), (80, 150, 80)),
        "_niche_score": 8.25,
    },
    {
        "title": "Nut Butter Too Expensive? Make Seed Butter for $2 in 3 Minutes",
        "pain_point": "Nut butter costs $8-12 per jar and you eat it daily?",
        "audiences": ["Budget vegans", "Nut-free families", "Meal preppers"],
        "steps": [
            "Roast 2 cups sunflower or pumpkin seeds (10 min at 350°F)",
            "Blend in food processor 3-5 min until creamy — add salt + honey optional",
        ],
        "result": "$2 homemade seed butter that tastes better than store-bought",
        "hashtags": ["#seedbutter", "#nutfree", "#budgetvegan", "#plantbased", "#mealprep"],
        "image_title": "Seed Butter for $2",
        "image_subtitle": "3-Minute Recipe",
        "image_steps": "Roast • Blend • Spread",
        "colors": ((100, 70, 20), (190, 140, 50)),
        "_niche_score": 8.25,
    },
    {
        "title": "Best AI App to Diagnose Sick Plants From a Photo — Free & Instant",
        "pain_point": "Plants keep dying and you don't know why?",
        "audiences": ["Plant parents", "Beginner gardeners", "Tech-curious"],
        "steps": [
            "Download PictureThis or Google Lens — snap a photo of the sick leaf",
            "AI identifies disease + gives treatment steps in 5 seconds",
        ],
        "result": "Save your plants with AI diagnosis — no gardening degree needed",
        "hashtags": ["#AIgardening", "#plantcare", "#plantdisease", "#techgarden", "#houseplants"],
        "image_title": "AI Plant Doctor",
        "image_subtitle": "Free Photo Diagnosis",
        "image_steps": "Snap • Scan • Treat",
        "colors": ((20, 80, 60), (40, 170, 130)),
        "_niche_score": 8.15,
    },
    {
        "title": "Compost on a Tiny Balcony Without Smell — Bokashi Method Explained",
        "pain_point": "Neighbors already complaining about your compost smell?",
        "audiences": ["Apartment composters", "Balcony gardeners", "Zero-waste beginners"],
        "steps": [
            "Get a bokashi bin ($20-30) — add food scraps + sprinkle bokashi bran after each layer",
            "Seal tight, drain liquid every 2 days (use as plant fertilizer), bury soil after 2 weeks",
        ],
        "result": "Zero-smell composting on ANY balcony — fermentation not rotting",
        "hashtags": ["#bokashi", "#composting", "#balconygarden", "#zerowaste", "#urbanfarming"],
        "image_title": "Balcony Composting",
        "image_subtitle": "Zero Smell — Bokashi",
        "image_steps": "Layer • Seal • Drain • Bury",
        "colors": ((50, 80, 30), (100, 160, 60)),
        "_niche_score": 8.15,
    },
    # ── Pain Point Packs (score 8.25) ──
    {
        "title": "Allergic to Nuts? 5 Plant-Based Fat Sources That Won't Kill You",
        "pain_point": "Allergic to nuts — what plant-based fat source can I use?",
        "audiences": ["Nut-allergy sufferers", "Plant-based beginners", "Parents of allergic kids"],
        "steps": [
            "Top 5 nut-free fats: sunflower seed butter, tahini, avocado, coconut cream, hemp seed oil",
            "Swap 1:1 in any recipe — seed butter in smoothies, tahini in dressings, coconut cream in curries",
        ],
        "result": "Full healthy fats without ANY nut exposure — allergy-safe daily diet",
        "hashtags": ["#nutfree", "#plantbased", "#foodallergy", "#seedbutter", "#healthyfats"],
        "image_title": "Nut-Free Fats",
        "image_subtitle": "5 Safe Alternatives",
        "image_steps": "Seeds • Tahini • Avocado • Coconut • Hemp",
        "colors": ((80, 100, 40), (160, 190, 80)),
        "_niche_score": 8.25,
    },
    {
        "title": "Always Tired on Plant-Based Diet? You're Missing THIS Mineral",
        "pain_point": "How to get enough iron on plant-based diet? Always tired.",
        "audiences": ["Tired vegans", "Plant-based athletes", "Women 20-40"],
        "steps": [
            "Iron-rich combos: lentils + vitamin C (lemon), spinach + bell pepper, pumpkin seeds + orange",
            "Cook in cast iron pan (adds 2-3mg iron per meal) — avoid tea/coffee 1hr before iron meals",
        ],
        "result": "Double your iron absorption — energy back in 7-14 days",
        "hashtags": ["#irondeficiency", "#plantbased", "#veganhealth", "#tiredvegan", "#nutrition"],
        "image_title": "Plant Iron Hack",
        "image_subtitle": "Stop Being Tired",
        "image_steps": "Lentils + Lemon • Cast Iron • No Coffee",
        "colors": ((100, 30, 30), (180, 60, 60)),
        "_niche_score": 8.25,
    },
    {
        "title": "Kids Allergic to Everything — 5 Nut-Free School Lunch Ideas Under $3",
        "pain_point": "Kids allergic to everything — nut-free school lunch ideas?",
        "audiences": ["Parents of allergic kids", "Budget families", "School lunch prep"],
        "steps": [
            "Monday: sunflower butter + jam wrap, Tuesday: hummus + veggie sticks, Wednesday: seed crackers + guac",
            "Thursday: coconut yogurt + granola (nut-free), Friday: rice balls + edamame — all under $3",
        ],
        "result": "5-day rotation every kid will actually eat — zero nuts, zero complaints",
        "hashtags": ["#nutfreelunch", "#schoollunch", "#foodallergy", "#kidsmeals", "#budgetfamily"],
        "image_title": "Nut-Free Lunches",
        "image_subtitle": "5 Days Under $3",
        "image_steps": "Mon-Fri Rotation",
        "colors": ((30, 80, 120), (60, 160, 220)),
        "_niche_score": 8.25,
    },
    {
        "title": "Zero Budget Garden — How to Grow Food With Literally $0",
        "pain_point": "No budget for garden supplies — 100% free setup possible?",
        "audiences": ["Broke gardeners", "Zero-waste", "Survival preppers"],
        "steps": [
            "Containers: milk jugs, egg cartons, toilet rolls. Soil: compost from food scraps (2-4 weeks)",
            "Free seeds: save from tomato/pepper/avocado. Regrow: green onion, celery, lettuce in water",
        ],
        "result": "Full kitchen garden from literal trash — $0 spent, food in 2-4 weeks",
        "hashtags": ["#freegarden", "#zerocost", "#regrowing", "#foodsecurity", "#upcyclegarden"],
        "image_title": "$0 Garden Setup",
        "image_subtitle": "Grow Food for FREE",
        "image_steps": "Trash → Containers → Food",
        "colors": ((40, 90, 40), (80, 180, 80)),
        "_niche_score": 8.25,
    },
    {
        "title": "Food Prices Insane? Grow Your Own Protein for Almost Nothing",
        "pain_point": "Food prices keep rising — how to grow my own protein?",
        "audiences": ["Budget families", "Plant-based eaters", "Self-sufficiency seekers"],
        "steps": [
            "Top 3 home-grown proteins: lentil sprouts (ready in 3 days), edamame (60 days), chickpea microgreens",
            "Invested: $2 bag of dried lentils = 30+ servings of sprouts. No garden needed, just a jar + water",
        ],
        "result": "Homegrown protein for pennies — sprouting jar on your counter beats $8/lb meat",
        "hashtags": ["#growprotein", "#sprouts", "#lentils", "#foodprices", "#selfsufficient"],
        "image_title": "Grow Your Protein",
        "image_subtitle": "$2 → 30 Servings",
        "image_steps": "Soak • Rinse • Sprout • Eat",
        "colors": ((60, 80, 30), (120, 160, 60)),
        "_niche_score": 8.25,
    },
    {
        "title": "Laid Off? How Homesteading Can Cut Your Food Bill by 60%",
        "pain_point": "Laid off — can homesteading actually save money?",
        "audiences": ["Recently laid off", "Career changers", "Budget survival"],
        "steps": [
            "Month 1: sprouts + microgreens ($5 startup) = save $40/month on greens",
            "Month 2: add fermentation (sauerkraut, kimchi) + bread baking → save another $30/month on groceries",
        ],
        "result": "Cut 60% of food bill within 2 months — homesteading IS a financial strategy",
        "hashtags": ["#laidoff", "#homesteading", "#savemoney", "#budgetliving", "#foodsecurity"],
        "image_title": "Homestead to Save $$$",
        "image_subtitle": "Cut 60% Food Bill",
        "image_steps": "Sprouts → Ferment → Bake → Save",
        "colors": ((80, 60, 20), (160, 120, 40)),
        "_niche_score": 8.25,
    },
    {
        "title": "Lost Everything — How to Restart a Garden From Absolute Zero",
        "pain_point": "Evicted or moved — how to restart my garden from nothing?",
        "audiences": ["People starting over", "New renters", "Urban displaced"],
        "steps": [
            "Day 1: save kitchen scraps (green onion, garlic, potato eyes). Day 3: regrow in water cups",
            "Week 2: find free containers (craigslist, neighborhood buy-nothing groups). Week 3: swap seeds locally",
        ],
        "result": "Functioning garden rebuilt from ZERO in 30 days — no money, just resourcefulness",
        "hashtags": ["#startingover", "#zerogarden", "#regrowing", "#resilience", "#urbangarden"],
        "image_title": "Garden From Zero",
        "image_subtitle": "Rebuild in 30 Days",
        "image_steps": "Save → Regrow → Find → Swap",
        "colors": ((50, 50, 80), (100, 100, 170)),
        "_niche_score": 8.25,
    },
    {
        "title": "Make CBD-Free Sleep Tea From Your Garden — 3 Herbs That Actually Work",
        "pain_point": "Can't sleep but don't want supplements or CBD?",
        "audiences": ["Insomniacs", "Natural remedy seekers", "Herb growers"],
        "steps": [
            "Grow these 3: chamomile (easy from seed), lemon balm (grows like weed), valerian root (perennial)",
            "Steep 1 tsp each in hot water 10 min before bed — drink 30 min before sleep",
        ],
        "result": "Homegrown sleep tea for free — works in 20 min without any supplements",
        "hashtags": ["#sleeptea", "#herbgarden", "#naturalremedy", "#insomnia", "#growherbs"],
        "image_title": "Garden Sleep Tea",
        "image_subtitle": "3 Herbs That Work",
        "image_steps": "Chamomile • Lemon Balm • Valerian",
        "colors": ((40, 30, 70), (80, 60, 150)),
        "_niche_score": 8.15,
    },
]

# Combine all packs
ALL_PACKS = PRE_WRITTEN_PACKS + NICHE_HUNTER_PACKS


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
                "overlay_hook": ImageFont.truetype(font_path, 44),
                "overlay_brand": ImageFont.truetype(font_path, 28),
            }
    default = ImageFont.load_default()
    return {"title": default, "subtitle": default, "body": default, "small": default, "brand": default, "overlay_hook": default, "overlay_brand": default}


def overlay_text_on_image(image_path: str, pack: dict) -> str:
    """Overlay clean micro-niche hook text on AI-generated photo.
    
    Adds:
    - Semi-transparent dark band at bottom ~30% of image
    - Topic hook/question in readable white text (not too big)
    - Brand watermark small at bottom
    - Hashtags line at very bottom
    
    This makes the image meaningful and on-topic instead of just a stock photo.
    """
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    
    # Create overlay layer
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    fonts = load_fonts()
    hook_font = fonts["overlay_brand"]  # 28pt — small, clean text on image
    brand_font = fonts["overlay_brand"]
    
    # Get the hook/question text — prefer hook from DB, then pain_point, then topic title
    hook_text = pack.get("_db_hook", "") or pack.get("pain_point", "") or pack.get("title", "")
    # Clean it up — remove markdown, keep it VERY short (max 60 chars)
    hook_text = hook_text.replace("**", "").replace("###", "").replace("##", "").replace("#", "").strip()
    if len(hook_text) > 60:
        hook_text = hook_text[:57].rsplit(" ", 1)[0] + "..."
    
    # NO hashtags on image — keep it clean, hashtags go in caption only
    brand_text = "@TheRikeRootStories"
    
    tag_font = brand_font  # 28pt
    
    # ── MINIMAL overlay — short hook + brand only, NO hashtags ──
    padding = 30
    line_spacing = 6
    
    # Wrap hook text (small font, short text)
    hook_lines = wrap_text(hook_text, hook_font, w - padding * 2, draw)
    hook_height = len(hook_lines) * (28 + line_spacing)
    
    # Total text block height — very compact
    brand_h = 22 + line_spacing
    total_text_h = hook_height + brand_h + padding * 2 + 10
    
    # Subtle semi-transparent gradient band at very bottom (narrow)
    band_top = h - total_text_h - 20
    for y in range(band_top, h):
        progress = (y - band_top) / (h - band_top)
        alpha = int(120 * min(progress * 1.8, 1.0))  # lighter (120 vs 160)
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))
    
    # ── Draw hook text (centered, white, small) ──
    y_cursor = band_top + padding + 10
    for line in hook_lines:
        draw.text((w // 2, y_cursor), line, fill=(255, 255, 255, 220),
                  font=hook_font, anchor="mt")
        y_cursor += 28 + line_spacing
    
    # ── Draw brand (centered, very subtle) ──
    y_cursor += 8
    draw.text((w // 2, y_cursor), brand_text, fill=(200, 200, 200, 120),
              font=brand_font, anchor="mt")
    
    # Composite overlay onto image
    result = Image.alpha_composite(img, overlay)
    result = result.convert("RGB")
    
    # Save back (overwrite)
    result.save(image_path, "JPEG", quality=95)
    return image_path


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

def _strip_markdown(text: str) -> str:
    """Strip markdown syntax and format for TikTok plain-text readability.
    
    TikTok does NOT render markdown. Literal ** ### etc. look broken.
    Keep emojis (they render fine).
    
    KEY INSIGHT: On TikTok, a single \\n gives NO visible gap.
    Only \\n\\n (double newline) creates a visible paragraph break.
    So we convert every logical paragraph/section break to \\n\\n.
    """
    import re
    
    # ── Step 1: Strip markdown syntax ──
    # Remove ### headers — keep text after them
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove **bold** markers — keep inner text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Remove *italic* markers — keep inner text  
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Remove __ underline markers
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # Remove backtick code markers
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # ── Step 2: Clean up each line ──
    lines = [line.strip() for line in text.split('\n')]
    
    # ── Step 3: Rebuild with proper paragraph spacing ──
    # Every non-empty line after another non-empty line gets a blank line between
    # EXCEPT consecutive bullet/dash lines (keep them grouped)
    result_lines = []
    for i, line in enumerate(lines):
        if not line:
            # Keep existing blank lines
            if not result_lines or result_lines[-1] != '':
                result_lines.append('')
            continue
        
        is_bullet = bool(re.match(r'^[-•–—]\s', line))
        prev_was_bullet = (result_lines and 
                          result_lines[-1] != '' and 
                          bool(re.match(r'^[-•–—]\s', result_lines[-1])))
        
        if result_lines and result_lines[-1] != '':
            # If both current and previous are bullets, keep them close (single newline)
            if is_bullet and prev_was_bullet:
                pass  # no blank line — bullets stay grouped
            else:
                # Add blank line between different types of content
                result_lines.append('')
        
        result_lines.append(line)
    
    text = '\n'.join(result_lines)
    
    # ── Step 4: Clean up excessive blank lines (max 1 blank = \n\n) ──
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # ── Step 5: TikTok/Publer line break fix ──
    # Publer/TikTok API strips pure blank lines and invisible Unicode.
    # PROVEN HACK: A line containing only a dot/period (.) creates a visible
    # paragraph break on TikTok. This is widely used by creators.
    text = text.replace('\n\n', '\n.\n')
    
    return text.strip()


def build_caption(pack: dict, location: str, season: str) -> str:
    """Build TikTok description: FULL content (markdown-stripped) + hashtags.
    
    For TikTok photo posts, the 'text' field IS the description.
    TikTok is PLAIN TEXT — no markdown rendering.
    We strip **, ###, etc. but keep emojis and clean structure.
    """
    content = pack.get("content_formatted", "")
    title = pack.get("title", "")
    hashtags = pack.get("hashtags", ["plantbased", "vegan", "healthyeating", "wellness", "tiktok"])
    
    # Strip markdown from title and content
    title = _strip_markdown(title)
    content = _strip_markdown(content)
    
    # Ensure all hashtags have # prefix
    tag_str = " ".join("#" + t.lstrip("#") for t in hashtags if t.strip())
    
    # Build: Title + Content + separator + Hashtags
    # Use dot-line (.) for paragraph breaks — proven TikTok hack
    DOT = '.'  # simple period on its own line = visible gap on TikTok
    parts = []
    if title:
        parts.append(title)
        parts.append(DOT)
    if content:
        parts.append(content)
        parts.append(DOT)
        parts.append('— — —')
        parts.append(DOT)
    parts.append(tag_str)
    
    full_text = '\n'.join(parts)
    
    # TikTok photo posts have ~4000 char limit for description
    # If too long, trim content but ALWAYS keep title + hashtags
    if len(full_text) > 4000:
        max_content_len = 4000 - len(title) - len(tag_str) - 10  # 10 for newlines
        if max_content_len > 500:
            content = content[:max_content_len].rsplit("\n", 1)[0]  # trim at last newline
            parts = []
            if title:
                parts.append(title)
                parts.append('.')
            parts.append(content)
            parts.append('.')
            parts.append(tag_str)
            full_text = '\n'.join(parts)
    
    return full_text


# ═══════════════════════════════════════════════════════════════
# MODE SELECTION — Pre-written pack or Gemini-generated
# ═══════════════════════════════════════════════════════════════

def get_content_pack(mode: str = "auto") -> dict:
    """
    Get a content pack.
    mode='prewritten' → pick from pre-written packs
    mode='gemini' → generate new with Gemini (legacy, may fail if key blocked)
    mode='ai_generate' → ★ NEW: Smart LLM cascade (GitHub Models → Perplexity → fallback) + self-review
    mode='ai_niche' → ★ NEW: AI-generate from niche_hunter.db top scores
    mode='niche_hunter' → use Micro-Niche Hunter scored questions (needs Gemini)
    mode='pain_point' → use pain-point questions (needs Gemini)
    mode='hunter_prewritten' → use pre-written packs from niche_hunter scores (NO Gemini needed)
    mode='auto' → weighted random (35% ai_generate, 25% hunter_prewritten, 20% ai_niche, 15% prewritten, 5% gemini)
    """
    if mode == "auto":
        mode = random.choices(
            ["ai_generate", "hunter_prewritten", "ai_niche", "prewritten", "gemini"],
            weights=[35, 25, 20, 15, 5],
            k=1,
        )[0]

    location = random.choice(LOCATIONS)
    season = random.choice(SEASONS)

    # ── Load already-published titles for deduplication ──
    published_titles = set()
    try:
        from web.app import get_db_safe, init_db
        init_db()
        with get_db_safe() as conn:
            rows = conn.execute(
                "SELECT title FROM posts WHERE status = 'published'"
            ).fetchall()
            published_titles = {r[0] for r in rows}
    except Exception:
        pass  # DB unavailable — skip dedup

    def _pick_unseen(candidates, top_n=None):
        """Pick a random pack not yet published. Falls back to any if all published."""
        pool = candidates[:top_n] if top_n else candidates
        unseen = [p for p in pool if p["title"] not in published_titles]
        if unseen:
            return random.choice(unseen)
        # All top picks already published — expand to full pool
        unseen_all = [p for p in candidates if p["title"] not in published_titles]
        return random.choice(unseen_all) if unseen_all else random.choice(pool)

    if mode == "hunter_prewritten":
        # Pick from niche_hunter scored packs — sorted by score descending
        sorted_packs = sorted(NICHE_HUNTER_PACKS, key=lambda p: p.get("_niche_score", 0), reverse=True)
        pack = _pick_unseen(sorted_packs, top_n=8)
        pack["_location"] = location
        pack["_season"] = season
        pack["_source"] = "hunter_prewritten"
        print(f"  Mode: Niche Hunter Pre-written (score={pack.get('_niche_score', 'N/A')})")
        print(f"  Title: {pack['title']}")
        return pack

    if mode == "prewritten":
        pack = _pick_unseen(ALL_PACKS)
        pack["_location"] = location
        pack["_season"] = season
        pack["_source"] = "prewritten"
        print(f"  Mode: Pre-written pack")
        print(f"  Title: {pack['title']}")
        return pack

    # ── AI Generate mode (Smart LLM Cascade + Self-Review) ──
    if mode == "ai_generate":
        print(f"  Mode: AI Generate (LLM Cascade + ReconcileGPT Review)")
        try:
            from llm_content import generate_content_pack
            # Pick a topic from micro-niches or niche_hunter DB
            topic_sources = MICRO_NICHES + NANO_NICHES + REAL_LIFE_NICHES
            topic = random.choice(topic_sources)
            pack = generate_content_pack(topic, score=7.5)
            if pack:
                pack["_location"] = location
                pack["_season"] = season
                # Add niche-specific hashtags from matrix
                niche_key = topic.split()[0].lower() if " " in topic else topic.lower()
                if niche_key in HASHTAG_MATRIX:
                    matrix = HASHTAG_MATRIX[niche_key]
                    extra_tags = matrix["micro"][:2] + matrix["nano"][:1] + matrix["trend"][:1]
                    existing = set(t.lower() for t in pack.get("hashtags", []))
                    for tag in extra_tags:
                        if tag.lower() not in existing:
                            pack["hashtags"].append(tag)
                            existing.add(tag.lower())
                print(f"  Review Score: {pack.get('_review_score', 'N/A')}/10")
                return pack
            else:
                print(f"  ⚠️ AI generation failed — falling back to hunter_prewritten")
                return get_content_pack("hunter_prewritten")
        except Exception as e:
            print(f"  ⚠️ AI Generate error: {e} — falling back to hunter_prewritten")
            return get_content_pack("hunter_prewritten")

    # ── AI Niche mode (Best niche_hunter topic + LLM cascade) ──
    if mode == "ai_niche":
        print(f"  Mode: AI Niche (niche_hunter.db top scores + LLM Cascade)")
        try:
            from llm_content import generate_from_niche_hunter
            pack = generate_from_niche_hunter(top_n=10)
            if pack:
                pack["_location"] = location
                pack["_season"] = season
                print(f"  Review Score: {pack.get('_review_score', 'N/A')}/10")
                return pack
            else:
                print(f"  ⚠️ AI Niche generation failed — falling back to hunter_prewritten")
                return get_content_pack("hunter_prewritten")
        except Exception as e:
            print(f"  ⚠️ AI Niche error: {e} — falling back to hunter_prewritten")
            return get_content_pack("hunter_prewritten")

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

def main(content_pack_override: dict = None):
    import httpx

    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    print("=" * 60)
    print(f"VIRALOPS MICRO-NICHE PUBLISHER (mode={mode})")
    print("=" * 60)

    # ── Step 1: Get content pack ──
    print("\n[1/5] Getting content pack...")
    if content_pack_override:
        pack = content_pack_override
        print(f"  ✓ Using provided content pack: {pack.get('title', 'N/A')}")
    else:
        pack = get_content_pack(mode)
    location = pack.get("_location", "Chicago")
    season = pack.get("_season", "Winter")

    # ── Step 2: Generate 9:16 image ──
    print("\n[2/5] Generating 9:16 TikTok image...")
    tmpdir = tempfile.mkdtemp(prefix="viralops_microniche_")
    
    # Check if pack has an AI-generated realistic image (from quality generator)
    ai_image = pack.get("_ai_image_path", "")
    if ai_image and os.path.exists(ai_image):
        image_path = ai_image
        print(f"  ✓ Using AI-generated realistic image: {image_path}")
    else:
        # Fallback: try generating AI image on the fly
        try:
            from llm_content import generate_image_for_pack
            ai_result = generate_image_for_pack(pack, tmpdir)
            if ai_result and os.path.exists(ai_result):
                image_path = ai_result
                print(f"  ✓ AI image generated on-the-fly: {image_path}")
            else:
                image_path = generate_post_image(pack, tmpdir)
                print(f"  ℹ Using PIL gradient image (AI image generation unavailable)")
        except Exception as e:
            image_path = generate_post_image(pack, tmpdir)
            print(f"  ℹ Using PIL gradient fallback: {e}")
    
    # ── Overlay clean micro-niche hook text on the image ──
    try:
        image_path = overlay_text_on_image(image_path, pack)
        print(f"  ✓ Text overlay added (hook + hashtags + brand)")
    except Exception as e:
        print(f"  ⚠ Text overlay failed (non-critical): {e}")
    
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

    publish_success = False
    publish_error = ""

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
                            publish_error = failure.get("message", str(failure))
                            print(f"\n  FAILED: {publish_error}")
                        elif post_info.get("state") in ("live", "published"):
                            post_link = post_info.get("post_link", "")
                            publish_success = True
                            print(f"\n  SUCCESS! Post is LIVE!")
                            print(f"  Link: {post_link}")
                        else:
                            post_link = post_info.get("post_link", "")
                            state = post_info.get("state", "unknown")
                            # scheduled/queued states are also success
                            if state in ("scheduled", "queued", "sending"):
                                publish_success = True
                            print(f"\n  State: {state}")
                            print(f"  Link: {post_link}")
                    break
            else:
                print(f"    [{attempt + 1}] HTTP {jr.status_code}")

    # ── Step 5: Save to DB ──
    print("\n[5/5] Saving to database...")
    db_status = "published" if publish_success else "failed"
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
                    db_status,
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
                        "error": publish_error,
                    }),
                ),
            )
            post_id = cur.lastrowid
            conn.execute(
                "INSERT INTO publish_log (post_id, platform, success, post_url, error) VALUES (?, ?, ?, ?, ?)",
                (post_id, "tiktok", 1 if publish_success else 0, post_link, publish_error),
            )
            conn.commit()
            print(f"  Saved as post #{post_id} (status={db_status})")
    except Exception as e:
        print(f"  DB save failed (non-critical): {e}")

    print("\n" + "=" * 60)
    if publish_success:
        print("DONE! Micro-niche content published to TikTok.")
    else:
        print(f"PUBLISH FAILED: {publish_error or 'unknown error'}")
    print(f"Title: {pack['title']}")
    print(f"Source: {pack.get('_source', 'unknown')}")
    print(f"Niche: {pack.get('_niche', 'prewritten')}")
    print("=" * 60)
    return publish_success


def batch_publish(count=3, gap_minutes=2):
    """Publish multiple posts with proper gaps. Waits between each post."""
    import httpx  # noqa: F811

    modes = ["hunter_prewritten", "prewritten"]
    print("=" * 60)
    print(f"VIRALOPS BATCH PUBLISHER — {count} posts, {gap_minutes}min gap")
    print("=" * 60)

    results = []
    for i in range(count):
        mode = modes[i % len(modes)]
        print(f"\n{'─' * 40}")
        print(f"Post {i + 1}/{count} (mode={mode})")
        print(f"{'─' * 40}")

        try:
            # Run a single publish
            sys.argv = ["publish_microniche.py", mode]
            success = main()
            results.append(("OK" if success else "FAIL", mode))
        except SystemExit:
            results.append(("EXIT", mode))
        except Exception as e:
            results.append(("ERROR", str(e)))

        if i < count - 1:
            wait_secs = gap_minutes * 60 + 15  # extra 15s safety margin
            print(f"\n  Waiting {wait_secs}s before next post...")
            time.sleep(wait_secs)

    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    for idx, (status, mode) in enumerate(results, 1):
        print(f"  Post {idx}: {status} ({mode})")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        batch_publish(count=count)
    else:
        main()
