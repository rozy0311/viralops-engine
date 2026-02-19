#!/usr/bin/env python3
"""
TikTok Micro-Niche Hunter  ViralOps Engine
=============================================
Architecture: EMADS-PR inspired (Training Multi-Agent)
Scoring: 4-Axis Weighted Formula
  Final = 0.35*Trend + 0.30*Demand + 0.25*LowComp + 0.10*LocalFit

Broad Niches:
  - plant-based raw material
  - howto-diy
  - AI green living lifestyle
  - AI Sustainability
  - homesteading

TikTok Channels Referenced:
  @therikerootstories   plant-based raw materials, micro-farming, DIY Chicago
  @agrinomadsvietnam    Vietnamese social commentary, homesteading, AI
  @therikecom           sustainable living, plant-based wellness, herbal medicine

Cost-Aware: Uses Gemini 2.5 Flash (budget-friendly) with local fallback
"""

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# 
# SECTION 1: SCORING DATA STRUCTURES (EMADS-PR Style)
# 

@dataclass
class NicheScore:
    """4-Axis Scoring for a micro-niche topic"""
    topic: str
    trend: float = 0.0       # 0-10: Google Trends / TikTok search volume
    demand: float = 0.0      # 0-10: Question frequency / search intent
    low_comp: float = 0.0    # 0-10: Low competition = high score
    local_fit: float = 0.0   # 0-10: Fit with Chicago / Vietnam / US context

    # Weighted formula
    WEIGHT_TREND = 0.35
    WEIGHT_DEMAND = 0.30
    WEIGHT_LOW_COMP = 0.25
    WEIGHT_LOCAL_FIT = 0.10

    @property
    def final_score(self) -> float:
        return round(
            self.WEIGHT_TREND * self.trend
            + self.WEIGHT_DEMAND * self.demand
            + self.WEIGHT_LOW_COMP * self.low_comp
            + self.WEIGHT_LOCAL_FIT * self.local_fit,
            2,
        )

    @property
    def risk_level(self) -> str:
        s = self.final_score
        if s >= 7:
            return " HIGH POTENTIAL"
        elif s >= 4:
            return " MODERATE"
        else:
            return " LOW  SKIP"

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "trend": self.trend,
            "demand": self.demand,
            "low_comp": self.low_comp,
            "local_fit": self.local_fit,
            "final_score": self.final_score,
            "risk_level": self.risk_level,
        }


@dataclass
class MicroNicheSeed:
    """Expanded micro-niche seed: [person] + [problem] + [result] + [local]"""
    person: str         # Who has this problem
    problem: str        # What they struggle with
    result: str         # Desired outcome
    local_context: str  # Local twist (Chicago, Vietnam, US)
    broad_niche: str    # Parent niche category
    hook: str = ""      # TikTok hook line
    hashtags: list = field(default_factory=list)
    series_name: str = ""
    monetize_angle: str = ""
    score: Optional[NicheScore] = None

    @property
    def seed_phrase(self) -> str:
        return f"{self.person} + {self.problem} + {self.result} + {self.local_context}"


# 
# SECTION 2: HIGHEST-SEARCH QUESTIONS DATABASE
# Pain points people desperately want answered
# 

HIGHEST_SEARCH_QUESTIONS = {
    "plant_based_raw_material": [
        #  Food Processing & DIY 
        {"q": "How to extract pectin from fruit peels at home?", "demand": 9, "trend": 8, "comp": 3, "local": 8},
        {"q": "How to make sunflower seed butter without a food processor?", "demand": 8, "trend": 7, "comp": 4, "local": 9},
        {"q": "How to sprout chickpeas for raw vegan protein?", "demand": 9, "trend": 8, "comp": 3, "local": 8},
        {"q": "DIY banana peel shoe polish  does it work?", "demand": 7, "trend": 9, "comp": 2, "local": 7},
        {"q": "How to make plant-based leather from mushroom mycelium?", "demand": 8, "trend": 9, "comp": 2, "local": 6},
        {"q": "Coffee grounds + eggshells fertilizer for tomatoes  real or myth?", "demand": 9, "trend": 8, "comp": 4, "local": 9},
        {"q": "How to make vinegar from fruit scraps zero waste?", "demand": 8, "trend": 7, "comp": 3, "local": 8},
        {"q": "Best raw plant-based protein sources for muscle building?", "demand": 9, "trend": 8, "comp": 5, "local": 7},
        {"q": "How to extract natural dyes from kitchen scraps?", "demand": 7, "trend": 8, "comp": 2, "local": 7},
        {"q": "How to make almond milk without waste  pulp recipes?", "demand": 8, "trend": 7, "comp": 5, "local": 8},
        #  Specific Plant Materials 
        {"q": "How to process raw hemp seeds for daily use?", "demand": 8, "trend": 8, "comp": 3, "local": 7},
        {"q": "How to make oat milk creamy like store-bought?", "demand": 9, "trend": 9, "comp": 6, "local": 8},
        {"q": "Chia seeds vs flax seeds  which is better for what?", "demand": 8, "trend": 7, "comp": 6, "local": 7},
        {"q": "How to ferment cashews for vegan cheese at home?", "demand": 8, "trend": 8, "comp": 3, "local": 7},
        {"q": "How to use banana fiber for DIY crafts?", "demand": 6, "trend": 8, "comp": 2, "local": 6},
    ],

    "howto_diy": [
        #  Urban Farming 
        {"q": "How to grow mushrooms in apartment with no sunlight?", "demand": 9, "trend": 9, "comp": 3, "local": 9},
        {"q": "How to build a self-watering planter from recycled bottles?", "demand": 8, "trend": 8, "comp": 3, "local": 9},
        {"q": "How to compost on a tiny balcony without smell?", "demand": 9, "trend": 8, "comp": 3, "local": 9},
        {"q": "How to grow microgreens with just a tray and window?", "demand": 9, "trend": 9, "comp": 4, "local": 9},
        {"q": "How to regrow green onions from kitchen scraps forever?", "demand": 8, "trend": 7, "comp": 5, "local": 9},
        {"q": "DIY raised bed from free pallets  step by step?", "demand": 8, "trend": 8, "comp": 4, "local": 8},
        {"q": "How to grow potatoes in a 5 gallon bucket?", "demand": 9, "trend": 9, "comp": 4, "local": 9},
        #  Home/Craft 
        {"q": "How to make natural soap from scratch without lye burns?", "demand": 8, "trend": 7, "comp": 3, "local": 7},
        {"q": "How to make beeswax wraps to replace plastic wrap?", "demand": 8, "trend": 8, "comp": 3, "local": 8},
        {"q": "How to fix cracked phone screen at home for cheap?", "demand": 7, "trend": 6, "comp": 7, "local": 6},
        {"q": "How to make natural candles from cooking oil leftover?", "demand": 7, "trend": 8, "comp": 2, "local": 7},
        {"q": "How to dry herbs for year-round use  oven vs air?", "demand": 8, "trend": 7, "comp": 4, "local": 8},
        {"q": "How to build a worm bin for vermicomposting?", "demand": 8, "trend": 7, "comp": 3, "local": 8},
        {"q": "How to make fire cider immune booster at home?", "demand": 7, "trend": 8, "comp": 3, "local": 7},
        {"q": "How to reuse potting soil safely next season?", "demand": 8, "trend": 7, "comp": 3, "local": 9},
    ],

    "ai_green_living": [
        #  AI + Gardening 
        {"q": "Best AI apps for identifying plant diseases from photos?", "demand": 9, "trend": 9, "comp": 4, "local": 8},
        {"q": "How to use ChatGPT to plan a garden layout?", "demand": 8, "trend": 9, "comp": 3, "local": 8},
        {"q": "AI tools that detect fake organic food labels?", "demand": 7, "trend": 8, "comp": 2, "local": 7},
        {"q": "How AI can reduce food waste at home?", "demand": 8, "trend": 8, "comp": 3, "local": 7},
        {"q": "AI-powered composting  smart sensors for optimal decomposition?", "demand": 6, "trend": 8, "comp": 2, "local": 6},
        {"q": "How to use AI to detect AI-generated fake news?", "demand": 9, "trend": 10, "comp": 5, "local": 7},
        {"q": "Best AI tools for sustainable shopping decisions?", "demand": 7, "trend": 8, "comp": 3, "local": 7},
        #  AI + Sustainability 
        {"q": "How AI helps optimize solar panel placement at home?", "demand": 7, "trend": 8, "comp": 3, "local": 7},
        {"q": "AI carbon footprint calculator  which one is accurate?", "demand": 7, "trend": 8, "comp": 3, "local": 7},
        {"q": "How to use Copilot/ChatGPT for meal prep with leftovers?", "demand": 8, "trend": 9, "comp": 4, "local": 8},
        {"q": "AI singing  how does it sound so real now?", "demand": 8, "trend": 9, "comp": 5, "local": 6},
        {"q": "Will AI replace human creativity?", "demand": 7, "trend": 8, "comp": 7, "local": 5},
    ],

    "ai_sustainability": [
        {"q": "How to use AI to track personal carbon emissions?", "demand": 7, "trend": 8, "comp": 3, "local": 7},
        {"q": "AI-powered water usage optimization for small farms?", "demand": 7, "trend": 8, "comp": 2, "local": 7},
        {"q": "How AI predicts crop diseases before they spread?", "demand": 8, "trend": 9, "comp": 3, "local": 7},
        {"q": "Smart home AI that reduces electricity bill by 30%?", "demand": 8, "trend": 8, "comp": 4, "local": 7},
        {"q": "AI tools for sustainable fashion  which brands are ethical?", "demand": 7, "trend": 8, "comp": 3, "local": 6},
        {"q": "How AI detects greenwashing in corporate sustainability reports?", "demand": 7, "trend": 9, "comp": 2, "local": 6},
        {"q": "AI precision farming  can small farmers afford it?", "demand": 8, "trend": 8, "comp": 3, "local": 7},
        {"q": "AI-generated content flooding TikTok  how to detect?", "demand": 9, "trend": 10, "comp": 4, "local": 7},
        {"q": "How AI can help restore degraded farmland?", "demand": 6, "trend": 7, "comp": 2, "local": 6},
        {"q": "AI + aquaponics  automated fish + plant ecosystem?", "demand": 7, "trend": 8, "comp": 2, "local": 7},
    ],

    "homesteading": [
        #  Urban Homesteading 
        {"q": "How to start homesteading in a small apartment?", "demand": 9, "trend": 9, "comp": 4, "local": 9},
        {"q": "Free materials for urban homesteading in Chicago?", "demand": 8, "trend": 7, "comp": 2, "local": 10},
        {"q": "How to preserve food without a canner  fermentation basics?", "demand": 8, "trend": 8, "comp": 3, "local": 8},
        {"q": "How to raise chickens on a balcony legally?", "demand": 7, "trend": 7, "comp": 3, "local": 7},
        {"q": "How to make herbal tinctures from garden herbs?", "demand": 8, "trend": 8, "comp": 3, "local": 8},
        {"q": "Best medicinal herbs to grow in pots  beginner list?", "demand": 9, "trend": 8, "comp": 4, "local": 8},
        {"q": "How to build a rain barrel from a trash can?", "demand": 7, "trend": 7, "comp": 3, "local": 8},
        {"q": "Cheapest way to heat a greenhouse in winter?", "demand": 8, "trend": 8, "comp": 4, "local": 8},
        #  Food Preservation 
        {"q": "How to dry mushrooms without a dehydrator?", "demand": 8, "trend": 7, "comp": 3, "local": 8},
        {"q": "How to make sourdough starter from scratch  day by day?", "demand": 9, "trend": 8, "comp": 5, "local": 7},
        {"q": "How to grow lemongrass indoors in cold climate?", "demand": 8, "trend": 8, "comp": 3, "local": 9},
        {"q": "Harden off seedlings  step by step for beginners?", "demand": 8, "trend": 8, "comp": 3, "local": 9},
        {"q": "How to grow water spinach (kangkong) in buckets?", "demand": 8, "trend": 8, "comp": 2, "local": 9},
        {"q": "Globe amaranth drying  how to keep color vibrant?", "demand": 7, "trend": 7, "comp": 2, "local": 8},
        {"q": "Sedum Autumn Joy for cooling urban courtyards?", "demand": 6, "trend": 7, "comp": 2, "local": 8},
    ],
}

# 
# SECTION 3: PAIN POINT DATABASE
# "Muốn được giải đáp ngay lập tức"  Immediate answers needed
# 

PAIN_POINT_QUESTIONS = {
    "plant_based_raw_material": [
        {"pain": "Nut butter too expensive  what's a cheap seed alternative?", "urgency": 10, "emotion": "frustration+budget"},
        {"pain": "Allergic to nuts  what plant-based fat source can I use?", "urgency": 10, "emotion": "health_fear"},
        {"pain": "Plant protein powder tastes disgusting  raw alternatives?", "urgency": 8, "emotion": "frustration"},
        {"pain": "My homemade oat milk separates  how to fix?", "urgency": 7, "emotion": "frustration"},
        {"pain": "Vegan cheese never melts right  cashew fermentation trick?", "urgency": 8, "emotion": "frustration"},
        {"pain": "Buying organic is bankrupting me  what to grow myself?", "urgency": 9, "emotion": "budget_panic"},
        {"pain": "How to get enough iron on plant-based diet? Always tired.", "urgency": 10, "emotion": "health_fear"},
        {"pain": "Kids allergic to everything  nut-free school lunch ideas?", "urgency": 10, "emotion": "parent_stress"},
    ],

    "howto_diy": [
        {"pain": "My apartment is too dark  can I still grow food?", "urgency": 9, "emotion": "hopelessness"},
        {"pain": "Compost bin stinks  neighbors complaining. What to do?", "urgency": 10, "emotion": "social_pressure"},
        {"pain": "Seedlings keep dying after transplant. What am I doing wrong?", "urgency": 9, "emotion": "frustration"},
        {"pain": "Potting soil from last year  safe to reuse or toss?", "urgency": 7, "emotion": "waste_guilt"},
        {"pain": "Mold growing on my mushroom kit. Is it ruined?", "urgency": 9, "emotion": "panic"},
        {"pain": "Balcony too small for garden  what CAN I grow?", "urgency": 8, "emotion": "space_frustration"},
        {"pain": "Tomato plants dying from bottom  brown spots on leaves?", "urgency": 9, "emotion": "panic"},
        {"pain": "No budget for garden supplies  100% free setup possible?", "urgency": 10, "emotion": "budget_desperation"},
    ],

    "ai_green_living": [
        {"pain": "Is this product actually eco-friendly or greenwashing?", "urgency": 8, "emotion": "distrust"},
        {"pain": "AI-generated articles everywhere  can't trust what I read", "urgency": 9, "emotion": "information_anxiety"},
        {"pain": "How to use AI without feeling like I'm cheating?", "urgency": 7, "emotion": "guilt"},
        {"pain": "AI tools cost too much  any free alternatives?", "urgency": 8, "emotion": "budget_frustration"},
        {"pain": "My plants keep dying  can AI diagnose from a photo?", "urgency": 9, "emotion": "desperation"},
        {"pain": "Electricity bill too high  AI suggestions to reduce?", "urgency": 9, "emotion": "budget_stress"},
    ],

    "ai_sustainability": [
        {"pain": "Carbon offset claims are confusing  which ones are real?", "urgency": 7, "emotion": "distrust"},
        {"pain": "My small farm can't compete with industrial  AI help?", "urgency": 9, "emotion": "existential_fear"},
        {"pain": "Food waste guilt  AI apps to plan meals from leftovers?", "urgency": 8, "emotion": "guilt"},
        {"pain": "Want to go solar but too expensive  AI calculators?", "urgency": 8, "emotion": "budget_barrier"},
        {"pain": "Are sustainable brands actually sustainable? AI fact-check?", "urgency": 7, "emotion": "distrust"},
    ],

    "homesteading": [
        {"pain": "Evicted  how to restart my garden from nothing?", "urgency": 10, "emotion": "crisis"},
        {"pain": "Winter is coming  how to keep plants alive indoors Chicago?", "urgency": 9, "emotion": "seasonal_panic"},
        {"pain": "Food prices keep rising  how to grow my own protein?", "urgency": 10, "emotion": "economic_fear"},
        {"pain": "I have zero gardening experience  where to even start?", "urgency": 9, "emotion": "overwhelm"},
        {"pain": "My sourdough starter died  what went wrong?", "urgency": 7, "emotion": "frustration"},
        {"pain": "HOA says no outdoor garden  stealth balcony options?", "urgency": 8, "emotion": "restriction_anger"},
        {"pain": "Herbs from store die in 3 days  how to keep them alive?", "urgency": 8, "emotion": "frustration"},
        {"pain": "Kid wants a pet but apartment rules  can we get worms?", "urgency": 6, "emotion": "creative_parenting"},
        {"pain": "Fire destroyed my home 17 months ago  rebuilding my sewing corner", "urgency": 10, "emotion": "recovery"},
        {"pain": "Laid off  can homesteading actually save money?", "urgency": 10, "emotion": "economic_survival"},
    ],
}


# 
# SECTION 4: MICRO-NICHE SEED EXPANSION ENGINE
# [person] + [problem] + [result] + [local context]
# 

MICRO_NICHE_SEEDS = [
    #  plant_based_raw_material 
    MicroNicheSeed(
        person="Budget vegan beginners",
        problem="nut butter too expensive, allergies",
        result="homemade sunflower seed butter in 3 minutes",
        local_context="Michigan/winter, Midwest prices",
        broad_niche="plant_based_raw_material",
        hook="Nut butter too expensive?  Try this $2 seed butter instead",
        hashtags=["#plantbased", "#veganrecipes", "#sunflowerbutter", "#budgetvegan", "#FoodTok"],
        series_name="$2 Plant-Based Staples",
        monetize_angle="Affiliate links to blenders/seeds on Amazon",
    ),
    MicroNicheSeed(
        person="Zero-waste home cooks",
        problem="fruit scraps going to trash",
        result="homemade vinegar from peels in 3 weeks",
        local_context="Chicago apartment, small kitchen",
        broad_niche="plant_based_raw_material",
        hook="Stop throwing fruit peels in the trash ",
        hashtags=["#zerowaste", "#fermentation", "#plantbased", "#DIYvinegar", "#sustainability"],
        series_name="Zero-Waste Kitchen Hacks",
        monetize_angle="Fermentation kit affiliate / e-book",
    ),
    MicroNicheSeed(
        person="Plant-based meal preppers",
        problem="protein sources are boring and expensive",
        result="7-day meal prep from chickpeas, microgreens, potatoes",
        local_context="Urban setup, small kitchen, under $30/week",
        broad_niche="plant_based_raw_material",
        hook="$30/week plant-based meal prep? Here's how ",
        hashtags=["#mealprep", "#plantbased", "#budgetfood", "#chickpeas", "#microgreens"],
        series_name="$30 Weekly Plant Prep",
        monetize_angle="Meal prep containers affiliate / recipe e-book",
    ),
    MicroNicheSeed(
        person="Nut-free school families",
        problem="kid allergic, every snack has nuts",
        result="seed-based snacks that are school-safe",
        local_context="US schools with nut bans",
        broad_niche="plant_based_raw_material",
        hook="Nut-free kid snacks that actually taste good ",
        hashtags=["#nutfree", "#schoollunch", "#allergyfriendly", "#kidssnacks", "#plantbased"],
        series_name="Nut-Free Kid Wins",
        monetize_angle="Seed butter brand partnerships",
    ),

    #  howto_diy 
    MicroNicheSeed(
        person="Apartment dwellers with no yard",
        problem="want to grow food but zero outdoor space",
        result="mushroom + microgreen setup in closet",
        local_context="Chicago apartment, $20 budget",
        broad_niche="howto_diy",
        hook="No yard? Grow food in your closet for $20 ",
        hashtags=["#urbanfarming", "#apartmentgarden", "#mushrooms", "#microgreens", "#DIY"],
        series_name="Closet Farm Series",
        monetize_angle="Mushroom kit + grow light affiliate links",
    ),
    MicroNicheSeed(
        person="First-time balcony gardeners",
        problem="seedlings keep dying after transplant",
        result="harden-off method that saves 95% of seedlings",
        local_context="Chicago zone 5b, spring timing",
        broad_niche="howto_diy",
        hook="Your seedlings die because you skip THIS step ",
        hashtags=["#gardeningtips", "#seedlings", "#hardenoff", "#chicagogarden", "#beginnergarner"],
        series_name="Don't Kill Your Plants",
        monetize_angle="Seed starting kit affiliate",
    ),
    MicroNicheSeed(
        person="Zero-budget homesteaders",
        problem="no money for garden supplies at all",
        result="100% free setup from city wood chips + food scraps",
        local_context="Chicago free wood chip program",
        broad_niche="howto_diy",
        hook="$0 garden setup? Chicago gives you free supplies ",
        hashtags=["#freegarden", "#urbanfarming", "#chicago", "#compost", "#woodchips"],
        series_name="$0 Homestead Hacks",
        monetize_angle="Local workshop promotions",
    ),
    MicroNicheSeed(
        person="Balcony herb beginners",
        problem="store herbs die in 3 days",
        result="5 herbs on 2m balcony, self-sustaining",
        local_context="Vietnam/HCM City, year-round growing",
        broad_niche="howto_diy",
        hook="Herbs from supermarket die in 3 days? Do THIS instead ",
        hashtags=["#herbgarden", "#balconygarden", "#growyourown", "#DIY", "#gardening"],
        series_name="Balcony Herb Lab",
        monetize_angle="Herb seed kit / Vietnamese audience affiliate",
    ),

    #  ai_green_living 
    MicroNicheSeed(
        person="Eco-conscious skeptics",
        problem="can't tell if AI content is real or fake",
        result="Pangram AI detector + media literacy skills",
        local_context="Global, TikTok content landscape",
        broad_niche="ai_green_living",
        hook="7% of news articles are AI-generated. Can you tell? ",
        hashtags=["#AI", "#aidetection", "#medialiteracy", "#fakenews", "#greenliving"],
        series_name="AI Truth Check",
        monetize_angle="Pangram affiliate / Chrome extension",
    ),
    MicroNicheSeed(
        person="Tech-curious gardeners",
        problem="plants keep dying, want AI diagnosis",
        result="free AI app scans leaf = instant diagnosis",
        local_context="Any location, smartphone required",
        broad_niche="ai_green_living",
        hook="Your phone can diagnose sick plants in 5 seconds ",
        hashtags=["#AIgardening", "#plantcare", "#plantdisease", "#tech", "#greenliving"],
        series_name="AI Plant Doctor",
        monetize_angle="Plant ID app referral / premium subscription",
    ),
    MicroNicheSeed(
        person="Budget-conscious homeowners",
        problem="electricity bill keeps going up",
        result="AI analyzes usage patterns, saves 30%",
        local_context="US residential, smart home optional",
        broad_niche="ai_green_living",
        hook="AI found $47/month I was wasting on electricity ",
        hashtags=["#savemoney", "#smartHome", "#AI", "#sustainability", "#energysaving"],
        series_name="AI Money Saver",
        monetize_angle="Smart plug / energy monitor affiliate",
    ),

    #  ai_sustainability 
    MicroNicheSeed(
        person="Conscious consumers",
        problem="brands claim eco-friendly but maybe lying",
        result="AI greenwashing detector reveals truth",
        local_context="Global brands, US market",
        broad_niche="ai_sustainability",
        hook="This brand claims 'eco-friendly'  AI says it's lying ",
        hashtags=["#greenwashing", "#sustainability", "#AI", "#ecoFriendly", "#truthcheck"],
        series_name="Greenwash Exposed",
        monetize_angle="Ethical brand partnerships",
    ),
    MicroNicheSeed(
        person="Small-scale farmers",
        problem="can't afford precision agriculture technology",
        result="free AI tools for crop disease prediction",
        local_context="Vietnam/US small farms",
        broad_niche="ai_sustainability",
        hook="Big farms use $50K AI. You can do this for FREE ",
        hashtags=["#farming", "#AI", "#agriculture", "#sustainability", "#smallfarm"],
        series_name="Free Farm AI",
        monetize_angle="AgTech tool reviews + consulting",
    ),

    #  homesteading 
    MicroNicheSeed(
        person="Newly displaced families",
        problem="lost home to fire/eviction, starting from zero",
        result="rebuild life skills: sewing, gardening, cooking",
        local_context="Chicago, post-displacement recovery",
        broad_niche="homesteading",
        hook="17 months displaced by fire. Here's how I'm rebuilding ",
        hashtags=["#homesteading", "#resilience", "#recovery", "#DIY", "#rebuild"],
        series_name="Rebuild From Zero",
        monetize_angle="GoFundMe partnerships, community resource guides",
    ),
    MicroNicheSeed(
        person="Urban apartment beginners",
        problem="want to homestead but live in a tiny apartment",
        result="mushrooms + herbs + microgreens + composting",
        local_context="Chicago apartment, winter-proof",
        broad_niche="homesteading",
        hook="Homesteading in a 500sqft apartment? YES you can ",
        hashtags=["#urbanhomesteading", "#apartmentgarden", "#chicago", "#mushrooms", "#herbs"],
        series_name="Apartment Homestead",
        monetize_angle="Kit bundles, workshop signups",
    ),
    MicroNicheSeed(
        person="Economic anxiety families",
        problem="food prices rising, income stagnant",
        result="grow protein + greens at home, save $200/month",
        local_context="US urban, Chicago specific",
        broad_niche="homesteading",
        hook="Food prices are insane. We saved $200/month growing our own ",
        hashtags=["#inflation", "#foodprices", "#growyourown", "#homesteading", "#savings"],
        series_name="Beat Inflation Garden",
        monetize_angle="Container garden kits, seed subscriptions",
    ),
    MicroNicheSeed(
        person="Vietnamese diaspora youth",
        problem="want to connect with roots through food",
        result="grow Vietnamese herbs in US apartment (tía tô, húng, sả)",
        local_context="US/Chicago, Vietnamese community",
        broad_niche="homesteading",
        hook="Grow Vietnamese herbs in your US apartment ",
        hashtags=["#VietnameseFood", "#herbgarden", "#diaspora", "#tíatô", "#homesteading"],
        series_name="Vietnamese Garden Abroad",
        monetize_angle="Vietnamese seed sources affiliate, cooking class",
    ),
    MicroNicheSeed(
        person="Medicinal herb enthusiasts",
        problem="buying herbal supplements is expensive and quality varies",
        result="grow + dry + prepare own lemongrass, ginkgo, senna, tulsi",
        local_context="Zone 5b-9, container growing",
        broad_niche="homesteading",
        hook="Stop buying expensive supplements  grow your medicine ",
        hashtags=["#herbalmedicine", "#lemongrass", "#holyBasil", "#homesteading", "#naturalhealth"],
        series_name="Grow Your Medicine",
        monetize_angle="Medicinal herb seed kits, herbal tincture workshop",
    ),
    MicroNicheSeed(
        person="Self-watering system builders",
        problem="plants die when traveling or busy",
        result="DIY wick-based self-watering planters from $5 buckets",
        local_context="Any apartment, minimal tools",
        broad_niche="homesteading",
        hook="Never kill another plant  $5 self-watering hack ",
        hashtags=["#selfwatering", "#DIY", "#gardeninghack", "#plantcare", "#homesteading"],
        series_name="Set It & Forget It Garden",
        monetize_angle="Self-watering planter kit affiliate",
    ),
]


# 
# SECTION 5: AUTO-SCORING ENGINE
# Scores micro-niche seeds using the 4-axis formula
# 

def score_question(q_data: dict) -> NicheScore:
    """Score a single question from the database"""
    return NicheScore(
        topic=q_data["q"] if "q" in q_data else q_data.get("pain", ""),
        trend=q_data.get("trend", 5),
        demand=q_data.get("demand", q_data.get("urgency", 5)),
        low_comp=10 - q_data.get("comp", 5),  # Invert: low comp = high score
        local_fit=q_data.get("local", 5),
    )


def score_all_questions() -> list[dict]:
    """Score every question in both databases, return sorted list"""
    results = []

    for niche, questions in HIGHEST_SEARCH_QUESTIONS.items():
        for q in questions:
            ns = score_question(q)
            r = ns.to_dict()
            r["niche"] = niche
            r["source"] = "search_question"
            results.append(r)

    for niche, pains in PAIN_POINT_QUESTIONS.items():
        for p in pains:
            ns = NicheScore(
                topic=p["pain"],
                trend=p.get("urgency", 5) * 0.8,  # Urgency approximates trend
                demand=p.get("urgency", 5),
                low_comp=7,  # Pain points usually low competition
                local_fit=7,
            )
            r = ns.to_dict()
            r["niche"] = niche
            r["source"] = "pain_point"
            r["emotion"] = p.get("emotion", "")
            results.append(r)

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def score_seed(seed: MicroNicheSeed) -> NicheScore:
    """Score a micro-niche seed based on its metadata"""
    # Find matching questions in DB to estimate scores
    niche_key = seed.broad_niche
    avg_trend = 7
    avg_demand = 7
    avg_comp = 3
    avg_local = 8

    if niche_key in HIGHEST_SEARCH_QUESTIONS:
        qs = HIGHEST_SEARCH_QUESTIONS[niche_key]
        if qs:
            avg_trend = sum(q["trend"] for q in qs) / len(qs)
            avg_demand = sum(q["demand"] for q in qs) / len(qs)
            avg_comp = sum(q.get("comp", 5) for q in qs) / len(qs)
            avg_local = sum(q.get("local", 5) for q in qs) / len(qs)

    return NicheScore(
        topic=seed.seed_phrase,
        trend=round(avg_trend, 1),
        demand=round(avg_demand, 1),
        low_comp=round(10 - avg_comp, 1),
        local_fit=round(avg_local, 1),
    )


def score_all_seeds() -> list[dict]:
    """Score all micro-niche seeds"""
    results = []
    for seed in MICRO_NICHE_SEEDS:
        ns = score_seed(seed)
        seed.score = ns
        r = ns.to_dict()
        r["niche"] = seed.broad_niche
        r["hook"] = seed.hook
        r["hashtags"] = seed.hashtags
        r["series"] = seed.series_name
        r["monetize"] = seed.monetize_angle
        r["person"] = seed.person
        results.append(r)

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# 
# SECTION 6: GEMINI AI EXPANSION (Cost-Aware)
# Uses Gemini 2.5 Flash for budget-friendly AI scoring
# 

def _get_gemini_api_keys() -> list[str]:
    """Return Gemini API keys in priority order (deduped).

    Supports:
      - GEMINI_API_KEY
      - FALLBACK_GEMINI_API_KEY
      - SECOND_FALLBACK_GEMINI_API_KEY
    """
    keys: list[str] = []
    for env_name in [
        "GEMINI_API_KEY",
        "FALLBACK_GEMINI_API_KEY",
        "SECOND_FALLBACK_GEMINI_API_KEY",
    ]:
        val = os.environ.get(env_name, "").strip()
        if val and val not in keys:
            keys.append(val)
    return keys


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def build_niche_expansion_prompt(broad_niche: str, count: int = 20) -> str:
    """Build prompt for Gemini to expand micro-niche seeds"""
    return f"""You are a TikTok Micro-Niche Hunter specializing in content that
goes viral in small, passionate communities.

BROAD NICHE: {broad_niche}

YOUR TASK: Generate {count} micro-niche content ideas that are:
1. Highly searchable questions people ASK on TikTok/Google
2. Pain points that need IMMEDIATE answers
3. Low competition (not covered by big creators)
4. Locally relevant to Chicago, Vietnam diaspora, or US urban audiences

FORMAT each idea as JSON:
{{
  "question": "The exact question people search for",
  "person": "Who has this problem (specific person type)",
  "problem": "Their specific pain point",
  "result": "What they want to achieve",
  "local_context": "Chicago/Vietnam/US specific angle",
  "trend_score": 0-10,
  "demand_score": 0-10,
  "competition": 0-10 (0=no competition, 10=saturated),
  "local_fit": 0-10,
  "hook": "TikTok hook line (first 3 seconds)",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"],
  "series_name": "Series this could belong to",
  "monetize": "How to make money from this content"
}}

NICHES TO FOCUS ON:
- plant-based raw materials (processing, DIY, alternatives)
- how-to DIY (urban farming, crafts, zero waste)
- AI in green living (AI tools for sustainability)
- homesteading (small space, urban, budget)

REFERENCE CHANNELS (match their style):
- @therikerootstories: educational plant-based how-to with citations
- @agrinomadsvietnam: Vietnamese commentary + homesteading tips
- @therikecom: deep research sustainable living + herbal medicine

Return a JSON array of {count} ideas. No markdown, just JSON."""


def expand_niches_with_gemini(broad_niche: str, count: int = 20) -> list[dict]:
    """Use Gemini to generate more micro-niche seeds"""
    try:
        from google import genai

        api_keys = _get_gemini_api_keys()
        if not api_keys:
            print("  No GEMINI_API_KEY(s)  using local database only")
            return []

        prompt = build_niche_expansion_prompt(broad_niche, count)
        last_err: Exception | None = None
        text = ""
        for idx, api_key in enumerate(api_keys, 1):
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                text = (response.text or "").strip()
                if text:
                    break
            except Exception as e:
                last_err = e
                # Common quota errors should fall through to next key
                print(f"  Gemini key {idx}/{len(api_keys)} failed: {str(e)[:120]}")
                continue

        if not text:
            raise last_err or RuntimeError("Gemini returned empty response")

        # Clean up JSON
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[: text.rfind("```")]

        ideas = json.loads(text)
        print(f" Gemini generated {len(ideas)} micro-niche ideas for '{broad_niche}'")
        return ideas

    except ImportError:
        print("  google-generativeai not installed  using local DB only")
        return []
    except Exception as e:
        print(f"  Gemini expansion failed: {e}")
        return []


def score_gemini_ideas(ideas: list[dict]) -> list[dict]:
    """Score Gemini-generated ideas using the 4-axis formula"""
    results = []
    for idea in ideas:
        ns = NicheScore(
            topic=idea.get("question", idea.get("hook", "")),
            trend=idea.get("trend_score", 5),
            demand=idea.get("demand_score", 5),
            low_comp=10 - idea.get("competition", 5),
            local_fit=idea.get("local_fit", 5),
        )
        r = ns.to_dict()
        r.update({
            "person": idea.get("person", ""),
            "hook": idea.get("hook", ""),
            "hashtags": idea.get("hashtags", []),
            "series": idea.get("series_name", ""),
            "monetize": idea.get("monetize", ""),
            "source": "gemini_expansion",
        })
        results.append(r)

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# 
# SECTION 7: TIKTOK CHANNEL STYLE MATCHER
# Maps content to the right channel based on style
# 

CHANNEL_PROFILES = {
    "therikerootstories": {
        "name": "The Rike Root Stories",
        "followers": 5677,
        "style": "educational_howto",
        "language": "English",
        "content_type": "plant-based raw materials, micro-farming, DIY recipes",
        "format": "photo posts with step-by-step + citations",
        "caption_style": "detailed educational with numbered steps, scientific refs",
        "location_focus": "Chicago, Michigan, US urban",
        "best_for": ["plant_based_raw_material", "howto_diy", "homesteading"],
        "example_posts": [
            "Sunflower seed butter in 3 minutes",
            "DIY pectin from fruit peels",
            "Chickpea sprout protein hacks",
            "Mushroom growing in apartment",
            "Microgreens + potatoes + beans urban setup",
            "Vietnamese perilla growing guide",
        ],
    },
    "agrinomadsvietnam": {
        "name": "Agrinomads Viet Nam",
        "followers": 12800,
        "style": "commentary_storytelling",
        "language": "Vietnamese",
        "content_type": "social commentary, homesteading, AI discussions, finance",
        "format": "photo + video posts, narrative storytelling",
        "caption_style": "conversational Vietnamese, emotional, provocative questions",
        "location_focus": "Vietnam, Vietnamese diaspora",
        "best_for": ["ai_green_living", "ai_sustainability", "homesteading"],
        "example_posts": [
            "Trồng nấm tại nhà siêu dễ cho newbie",
            "Trồng ớt, sả, gừng, húng trên ban công",
            "AI hát nghe ngon quá  cách hoạt động",
            "Khởi nghiệp  thực tế vs ảo tưởng",
            "Mạng xã hội tràn ngập AI  tỉnh táo",
        ],
    },
    "therikecom": {
        "name": "The Rike Stories",
        "followers": 6856,
        "style": "deep_research_educational",
        "language": "English",
        "content_type": "sustainable living, herbal medicine, plant science, AI detection",
        "format": "photo posts with 'Direct Answer' + Key Conditions + Understanding",
        "caption_style": "research-paper style with Direct Answer, Key Conditions, Understanding sections",
        "location_focus": "US, Chicago, global sustainability",
        "best_for": ["ai_green_living", "ai_sustainability", "homesteading"],
        "example_posts": [
            "AI detection with Pangram",
            "Self-watering planters  wick height optimization",
            "Water spinach semi-hydroponic buckets",
            "Kangkong continuous cut system",
            "Lemongrass benefits uses dosage",
            "Ginkgo biloba  home growing + medicine",
            "Sedum Autumn Joy for urban cooling",
            "Globe amaranth drying for vibrant color",
            "Hardening off seedlings step by step",
            "Cottage garden ideas for small yards",
        ],
    },
}


def match_channel(niche: str, language: str = "english") -> str:
    """Match a niche to the best TikTok channel"""
    if language.lower() in ["vietnamese", "vi", "vn"]:
        return "agrinomadsvietnam"

    best_match = "therikerootstories"  # default
    for channel_id, profile in CHANNEL_PROFILES.items():
        if niche in profile["best_for"]:
            if profile["language"].lower() == language.lower():
                best_match = channel_id
                break
            elif profile["language"] == "English":
                best_match = channel_id

    return best_match


# 
# SECTION 8: AUTOMATION COMPLEXITY SCORING (EMADS-PR)
# 

def automation_score() -> dict:
    """Calculate Automation Complexity Score for Niche Hunter (0-12)"""
    return {
        "data_sources": 3,  # TikTok search, Gemini AI, local question DB
        "logic_complexity": 2,  # 4-axis scoring + weighted formula (moderate)
        "integration_points": 2,  # Gemini API + Publer API
        "total": 7,
        "risk_level": " MEDIUM RISK",
        "action": "Requires content review before publish, staging test recommended",
    }


# 
# SECTION 9: OUTPUT  TOP 20 MICRO-NICHES TABLE
# 

def print_top_niches(results: list[dict], top_n: int = 20, title: str = ""):
    """Print formatted table of top micro-niches"""
    print(f"\n{'='*90}")
    print(f"  {title or 'TOP MICRO-NICHES'} (Top {top_n})")
    print(f"{'='*90}")
    print(f"{'#':>3} {'Score':>6} {'Level':^20} {'Niche':^25} {'Topic/Question'}")
    print(f"{'-'*3:>3} {'-'*6:>6} {'-'*20:^20} {'-'*25:^25} {'-'*40}")

    for i, r in enumerate(results[:top_n], 1):
        score = r["final_score"]
        level = r["risk_level"]
        niche = r.get("niche", "")[:24]
        topic = r["topic"][:60]
        print(f"{i:>3} {score:>6.2f} {level:^20} {niche:^25} {topic}")

    print(f"{'='*90}\n")


def print_content_calendar(results: list[dict], days: int = 7):
    """Generate a content calendar from top results"""
    print(f"\n{'='*90}")
    print(f"  CONTENT CALENDAR  Next {days} Days")
    print(f"{'='*90}")

    channels_used = set()
    for i, r in enumerate(results[:days], 1):
        niche = r.get("niche", "plant_based_raw_material")
        channel = match_channel(niche)
        channels_used.add(channel)

        print(f"\n  Day {i}: @{channel}")
        print(f"  Topic: {r['topic'][:70]}")
        print(f"  Score: {r['final_score']:.2f} {r['risk_level']}")
        if r.get("hook"):
            print(f"  Hook:  {r['hook']}")
        if r.get("hashtags"):
            tags = r["hashtags"]
            if isinstance(tags, list):
                print(f"  Tags:  {' '.join(tags[:5])}")
        if r.get("series"):
            print(f"  Series: {r['series']}")
        if r.get("monetize"):
            print(f"   Monetize: {r['monetize']}")

    print(f"\n{'='*90}")
    print(f" Channels: {', '.join(f'@{c}' for c in channels_used)}")
    print(f"{'='*90}\n")


# 
# SECTION 10: SQLITE PERSISTENCE
# 

def init_db(db_path: str = "niche_hunter.db"):
    """Initialize SQLite database for niche scores"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS niche_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            niche TEXT,
            trend REAL,
            demand REAL,
            low_comp REAL,
            local_fit REAL,
            final_score REAL,
            risk_level TEXT,
            source TEXT,
            hook TEXT,
            hashtags TEXT,
            series TEXT,
            monetize TEXT,
            channel TEXT,
            scored_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_scores(conn: sqlite3.Connection, results: list[dict]):
    """Save scored results to SQLite"""
    c = conn.cursor()
    for r in results:
        niche = r.get("niche", "")
        channel = match_channel(niche)
        hashtags = json.dumps(r.get("hashtags", []))
        c.execute("""
            INSERT INTO niche_scores
            (topic, niche, trend, demand, low_comp, local_fit,
             final_score, risk_level, source, hook, hashtags, series, monetize, channel)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["topic"], niche, r["trend"], r["demand"],
            r["low_comp"], r["local_fit"], r["final_score"],
            r["risk_level"], r.get("source", ""), r.get("hook", ""),
            hashtags, r.get("series", ""), r.get("monetize", ""), channel,
        ))
    conn.commit()
    print(f" Saved {len(results)} scores to database")


# 
# SECTION 11: MAIN WORKFLOW
# Input  Expand Seeds  Score  Rank  Output Top 20
# 

def run_niche_hunter(
    use_gemini: bool = True,
    gemini_count: int = 20,
    top_n: int = 20,
    save_db: bool = True,
    broad_niches: list[str] = None,
):
    """
    Full Niche Hunter Workflow (EMADS-PR inspired):
    1. Load local question + pain point databases
    2. Score all local seeds using 4-axis formula
    3. (Optional) Expand with Gemini AI per broad niche
    4. Score Gemini results
    5. Merge, deduplicate, rank top N
    6. Print results table + content calendar
    7. Save to SQLite
    """
    print("\n" + "="*90)
    print("  TikTok MICRO-NICHE HUNTER v1.0  ViralOps Engine")
    print(" Architecture: EMADS-PR | Scoring: 4-Axis Weighted")
    print(" Formula: 0.35*Trend + 0.30*Demand + 0.25*LowComp + 0.10*LocalFit")
    print("="*90)

    # Step 0: Automation Score
    auto = automation_score()
    print(f"\n Automation Score: {auto['total']}/12  {auto['risk_level']}")
    print(f"   {auto['action']}")

    # Use all niches if none specified
    if broad_niches is None:
        broad_niches = list(HIGHEST_SEARCH_QUESTIONS.keys())

    all_results = []

    # Step 1: Score local question database
    print(f"\n Step 1: Scoring {sum(len(v) for v in HIGHEST_SEARCH_QUESTIONS.values())} search questions...")
    q_results = score_all_questions()
    all_results.extend(q_results)
    print(f"    {len(q_results)} questions scored")

    # Step 2: Score micro-niche seeds
    print(f"\n Step 2: Scoring {len(MICRO_NICHE_SEEDS)} micro-niche seeds...")
    seed_results = score_all_seeds()
    all_results.extend(seed_results)
    print(f"    {len(seed_results)} seeds scored")

    # Step 3: Gemini AI expansion (cost-aware)
    if use_gemini and GEMINI_API_KEY:
        for niche in broad_niches:
            print(f"\n Step 3: Gemini expanding '{niche}' ({gemini_count} ideas)...")
            ideas = expand_niches_with_gemini(niche, gemini_count)
            if ideas:
                gemini_scored = score_gemini_ideas(ideas)
                all_results.extend(gemini_scored)
                print(f"    {len(gemini_scored)} Gemini ideas scored")
            time.sleep(1)  # Rate limiting
    else:
        print("\n Step 3: Skipping Gemini (no API key or disabled)")

    # Step 4: Sort and deduplicate
    all_results.sort(key=lambda x: x["final_score"], reverse=True)

    # Deduplicate by topic similarity (simple exact match)
    seen = set()
    unique_results = []
    for r in all_results:
        key = r["topic"].lower().strip()[:50]
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    print(f"\n TOTAL: {len(unique_results)} unique micro-niches scored")

    # Step 5: Display results
    print_top_niches(unique_results, top_n, "HIGHEST-SCORING MICRO-NICHES")

    # Per-niche breakdown
    for niche in broad_niches:
        niche_results = [r for r in unique_results if r.get("niche") == niche]
        if niche_results:
            print_top_niches(niche_results, 5, f"TOP 5: {niche.upper()}")

    # Content calendar
    top_with_hooks = [r for r in unique_results if r.get("hook")]
    if top_with_hooks:
        print_content_calendar(top_with_hooks, 7)

    # Step 6: Save to database
    if save_db:
        conn = init_db()
        save_scores(conn, unique_results)
        conn.close()

    return unique_results


# 
# SECTION 12: INTEGRATION WITH publish_microniche.py
# 

def get_top_content_pack(niche: str = None, top_n: int = 1) -> list[dict]:
    """Get top-scored content packs for publish_microniche.py integration"""
    results = score_all_questions()

    if niche:
        results = [r for r in results if r.get("niche") == niche]

    packs = []
    for r in results[:top_n]:
        pack = {
            "mode": "niche_hunter",
            "topic": r["topic"],
            "niche": r.get("niche", "plant_based_raw_material"),
            "score": r["final_score"],
            "channel": match_channel(r.get("niche", "")),
            "hook": r.get("hook", ""),
            "hashtags": r.get("hashtags", []),
            "series": r.get("series", ""),
            "monetize": r.get("monetize", ""),
        }
        packs.append(pack)

    return packs


def get_pain_point_content(emotion: str = None, top_n: int = 5) -> list[dict]:
    """Get pain-point content sorted by urgency for immediate-answer posts"""
    results = []
    for niche, pains in PAIN_POINT_QUESTIONS.items():
        for p in pains:
            if emotion and p.get("emotion") != emotion:
                continue
            ns = NicheScore(
                topic=p["pain"],
                trend=p.get("urgency", 5) * 0.8,
                demand=p.get("urgency", 5),
                low_comp=7,
                local_fit=7,
            )
            r = ns.to_dict()
            r["niche"] = niche
            r["emotion"] = p.get("emotion", "")
            r["channel"] = match_channel(niche)
            results.append(r)

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_n]


# 
# ENTRY POINT
# 

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "full":
        run_niche_hunter(use_gemini=True, top_n=20)
    elif mode == "local":
        run_niche_hunter(use_gemini=False, top_n=20)
    elif mode == "pain":
        results = get_pain_point_content(top_n=10)
        print_top_niches(results, 10, "TOP PAIN POINT CONTENT")
    elif mode == "calendar":
        results = run_niche_hunter(use_gemini=False, top_n=7, save_db=False)
        # Calendar already printed in run_niche_hunter
    elif mode == "seeds":
        results = score_all_seeds()
        print_top_niches(results, len(results), "ALL MICRO-NICHE SEEDS")
    else:
        print(f"Usage: python niche_hunter.py [full|local|pain|calendar|seeds]")