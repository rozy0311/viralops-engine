"""
Insert 40 fresh nano-niche v5 topics into niche_hunter.db.
Run once: python add_v5_topics.py
"""
import sqlite3
import os

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJ_DIR, "niche_hunter.db")

V5_TOPICS = [
    # ── Budget Kitchen Hacks ──
    ("Frozen banana ice cream 3 flavors no sugar no dairy blender 2 minutes kids love it", "howto_diy", 9.4,
     "Healthy dessert that costs pennies — just frozen bananas and a blender", "budget_food"),
    ("Rice cooker meals beyond rice steamed vegetables eggs cake one pot no stove needed", "howto_diy", 9.3,
     "Your 20 dollar rice cooker replaces half your kitchen", "kitchen_hack"),
    ("Stale bread 5 recipes french toast croutons breadcrumbs bread pudding zero food waste", "howto_diy", 9.3,
     "Never throw away stale bread again — 5 recipes in 10 minutes", "zero_waste"),
    ("Homemade yogurt from 2 ingredients milk and starter no machine just a towel 8 hours", "howto_diy", 9.3,
     "Homemade yogurt costs 30 cents vs 5 dollars at the store", "budget_food"),
    ("Air fryer frozen vegetables crispy in 8 minutes no oil needed taste like roasted", "howto_diy", 9.2,
     "Frozen veggies nobody wants become addictive in the air fryer", "kitchen_hack"),
    ("Leftover rice fried rice 5 minutes soy sauce egg green onion better than takeout", "howto_diy", 9.2,
     "Yesterday rice plus one egg equals better than restaurant fried rice", "budget_food"),
    ("Pasta water save it for plants bread sauce thickener 4 uses most people dump drain", "howto_diy", 9.2,
     "Stop pouring liquid gold down the drain — 4 uses for pasta water", "zero_waste"),
    ("Canned chickpea aquafaba whip meringue vegan egg replacement free from trash liquid", "howto_diy", 9.2,
     "The liquid you throw away from canned chickpeas is a miracle ingredient", "kitchen_hack"),

    # ── Urban Gardening & Plants ──
    ("Green onion regrow in water glass from grocery store infinite supply never buy again", "urban_garden", 9.4,
     "Buy green onions once and never again — just a glass of water", "garden_hack"),
    ("Avocado pit grow tree from seed toothpick water glass to pot 6 month journey", "urban_garden", 9.2,
     "From your breakfast avocado pit to a beautiful indoor tree — free", "garden_hack"),
    ("Compost bin apartment balcony small bucket no smell bokashi method 2 weeks soil ready", "urban_garden", 9.2,
     "Composting in a tiny apartment without any smell — the bokashi method", "zero_waste"),
    ("Potato grow bag on balcony 1 potato becomes 10 harvest 3 months grocery bag method", "urban_garden", 9.3,
     "Turn 1 potato into 10 on your apartment balcony — just need a bag", "garden_hack"),
    ("Mint propagation from single stem water glass roots 5 days unlimited mojito supply", "urban_garden", 9.2,
     "One mint stem from the store gives you unlimited fresh mint forever", "garden_hack"),
    ("Microgreens grow on kitchen counter paper towel seeds 7 days superfood 50 cents tray", "urban_garden", 9.3,
     "Grow superfoods on your counter for 50 cents in 7 days", "garden_hack"),
    ("Eggshell fertilizer crush dry sprinkle calcium boost for tomatoes peppers free nutrient", "urban_garden", 9.1,
     "Free fertilizer hiding in your breakfast — crushed eggshells for plants", "zero_waste"),
    ("Spider plant propagate babies separate repot 20 new plants from 1 mother air purifier", "urban_garden", 9.1,
     "One spider plant becomes 20 free air purifiers for your home", "garden_hack"),

    # ── Health & Wellness Budget ──
    ("Apple cider vinegar morning routine 1 tablespoon water digestion skin 30 day results", "health_wellness", 9.2,
     "The 3 dollar bottle that changed my morning routine — 30 day results", "wellness_hack"),
    ("Cold shower benefits 30 seconds end of shower immunity mood energy no equipment free", "health_wellness", 9.1,
     "30 seconds of cold water — the free health hack backed by science", "wellness_hack"),
    ("Turmeric golden milk recipe before bed anti inflammatory 4 ingredients 5 minutes sleep", "health_wellness", 9.2,
     "This bedtime drink costs 20 cents and fights inflammation while you sleep", "wellness_hack"),
    ("Magnesium foot soak epsom salt before bed 1 dollar per soak better sleep muscle recovery", "health_wellness", 9.1,
     "1 dollar foot soak before bed changed my sleep quality completely", "wellness_hack"),
    ("Ginger lemon honey tea homemade cold remedy immune boost 3 ingredients costs nothing", "health_wellness", 9.2,
     "Grandma was right — this 3 ingredient tea beats expensive cold medicine", "wellness_hack"),
    ("Morning stretching 5 minutes no equipment back pain relief office worker desk job routine", "health_wellness", 9.1,
     "5 minutes every morning fixed my desk job back pain — no gym needed", "wellness_hack"),

    # ── DIY & Home Hacks ──
    ("Baking soda vinegar drain cleaner no chemicals 2 ingredients unclog sink 10 minutes", "howto_diy", 9.3,
     "Unclog any drain with 2 kitchen ingredients — save 50 dollars on plumber", "home_hack"),
    ("White vinegar fabric softener replacement no chemicals 1 cup wash cycle clothes softer", "howto_diy", 9.1,
     "Replace 8 dollar fabric softener with 1 cup of white vinegar", "home_hack"),
    ("Dryer balls wool replace dryer sheets save 50 dollars year less drying time reusable", "howto_diy", 9.1,
     "These 10 dollar wool balls replace dryer sheets forever — save 50 per year", "home_hack"),
    ("Lemon microwave cleaner 5 minutes steam no scrub cut lemon water bowl deodorize", "howto_diy", 9.2,
     "5 minutes and half a lemon — microwave looks brand new no scrubbing", "home_hack"),
    ("Mason jar meal prep salad 5 jars 30 minutes sunday fresh by friday no soggy lettuce", "howto_diy", 9.3,
     "5 mason jar salads on Sunday — still fresh and crispy by Friday", "meal_prep"),
    ("Cast iron skillet restore rusty thrift store find 3 dollar pan better than nonstick forever", "howto_diy", 9.2,
     "This 3 dollar thrift store find replaced my 200 dollar pan set", "home_hack"),

    # ── Budget Lifestyle & Finance ──
    ("Library card free resources beyond books movies audiobooks wifi museum passes language apps", "budget_life", 9.2,
     "Your free library card unlocks 500 dollars worth of services most people ignore", "budget_hack"),
    ("Meal plan template weekly 50 dollars family of 4 grocery list seasonal produce bulk buy", "budget_life", 9.3,
     "Feed a family of 4 for 50 dollars a week — the exact meal plan template", "budget_food"),
    ("Cashback apps stack grocery shopping 3 apps 15 percent back Ibotta Fetch receipts", "budget_life", 9.1,
     "Stacking 3 free apps saves 15 percent on groceries you already buy", "budget_hack"),
    ("Freezer inventory list reduce food waste save 100 dollars month know what you have", "budget_life", 9.2,
     "A simple freezer inventory saved our family 100 dollars a month in food waste", "budget_hack"),
    ("Bulk cooking beans lentils rice pressure cooker 1 hour 10 meals freezer storage 3 months", "budget_life", 9.3,
     "1 hour of cooking on Sunday makes 10 freezer meals for under 10 dollars", "meal_prep"),

    # ── Sustainable Living ──
    ("Beeswax wrap make at home replace plastic wrap cotton fabric beeswax iron 10 minutes", "sustainability", 9.2,
     "Make your own beeswax wraps — ditch plastic wrap forever for 5 dollars", "zero_waste"),
    ("Reusable produce bags mesh drawstring grocery shopping no more plastic bags 5 dollar set", "sustainability", 9.0,
     "These 5 dollar bags eliminated all our plastic bag waste from groceries", "zero_waste"),
    ("Soap bar vs liquid body wash cost comparison 3 months plastic waste reduction numbers", "sustainability", 9.1,
     "Switching to bar soap saves 40 dollars a year and 12 plastic bottles", "zero_waste"),
    ("Cloth napkins replace paper towels family of 4 saves 200 dollars year zero waste kitchen", "sustainability", 9.1,
     "Paper towels cost 200 dollars per year — cloth napkins pay for themselves in 2 months", "zero_waste"),
    ("Vinegar weed killer garden safe for kids pets no roundup spray bottle sunshine 24 hours", "sustainability", 9.1,
     "This kitchen ingredient kills weeds without poisoning your kids or pets", "garden_hack"),

    # ── Quick Wins ──
    ("Ice cube tray uses beyond ice herbs olive oil coffee baby food portions leftover sauce", "howto_diy", 9.1,
     "8 genius uses for ice cube trays that have nothing to do with ice", "kitchen_hack"),
    ("Shower cap reuse covers bowls proofing dough shoe covers free hotel amenity hack", "howto_diy", 9.0,
     "Hotel shower caps are the most versatile free hack nobody talks about", "home_hack"),
]

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Check existing topics to avoid duplicates
    existing = set()
    for row in cur.execute("SELECT topic FROM niche_scores"):
        existing.add(row[0].lower().strip()[:50])

    inserted = 0
    skipped = 0
    for topic, niche, score, hook, channel in V5_TOPICS:
        key = topic.lower().strip()[:50]
        if key in existing:
            print(f"  SKIP (dup): {topic[:60]}...")
            skipped += 1
            continue

        cur.execute(
            """INSERT INTO niche_scores
               (topic, niche, trend, demand, low_comp, local_fit,
                final_score, risk_level, source, hook, channel)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                topic,
                niche,
                round(score * 0.9, 1),   # trend
                round(score * 0.95, 1),   # demand
                round(score * 0.85, 1),   # low_comp
                round(score * 0.8, 1),    # local_fit
                score,                     # final_score
                " HIGH POTENTIAL" if score >= 9.0 else " MODERATE",
                "nano_niche_v5",
                hook,
                "therikerootstories",
            ),
        )
        inserted += 1
        existing.add(key)

    conn.commit()

    # Verify
    total = cur.execute("SELECT COUNT(*) FROM niche_scores").fetchone()[0]
    v5_count = cur.execute("SELECT COUNT(*) FROM niche_scores WHERE source = 'nano_niche_v5'").fetchone()[0]
    conn.close()

    print(f"\n  Inserted: {inserted}")
    print(f"  Skipped:  {skipped}")
    print(f"  v5 total: {v5_count}")
    print(f"  DB total: {total}")

if __name__ == "__main__":
    main()
