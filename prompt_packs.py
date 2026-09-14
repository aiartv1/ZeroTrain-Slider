"""Concepts, scene generators and preservation sets. Plain data — edit freely.

Two rules are baked into this file, and both were learned the hard way:

**A scene must never mention the thing the slider controls.** The generator crosses
subject x context x framing x look, and for a camera-distance slider the ``framing``
axis emits "close-up" and "wide shot" — the two poles of the concept. The base prompt
then contradicts the concept phrase appended to it, the measured direction is noise, and
the slider comes out dead. Each concept therefore declares ``hold_axes``: the axes the
generator must leave *out* of its prompts. Age holds nothing (no axis mentions age);
camera distance holds ``framing``; style holds ``look``.

**A preservation set must never cover the concept's own domain.** Preservation prompts
enter the same equations with the opposite target, so preserving framings while building
a framing slider tells the solve to move and not move the same thing. Each concept
therefore declares which preservation set actually complements it: subject-attribute
sliders (age, weight) preserve scenes and objects; global-look sliders (detail, camera,
colour) preserve bare subjects.
"""

import random

# --------------------------------------------------------------------------------------
# concepts
# --------------------------------------------------------------------------------------
# hold_axes  : generator axes that would collide with this concept -> left out entirely.
# preserve   : preservation set that complements this concept rather than fighting it.

CONCEPTS = {
    "Age (young <-> old)": {
        "positive": "elderly, aged, deeply wrinkled skin, grey thinning hair, sagging features",
        "negative": "very young, youthful, smooth unlined skin, fresh and unaged features",
        "scenes": "people_portrait", "hold_axes": (), "preserve": "scenes_and_objects",
    },
    "Body weight (slim <-> heavy)": {
        "positive": "heavyset, overweight, full round body, thick torso and limbs, soft heavy build",
        "negative": "slim, slender, lean thin body, narrow frame, low body fat",
        "scenes": "people_full_body", "hold_axes": (), "preserve": "scenes_and_objects",
    },
    "Muscularity (soft <-> muscular)": {
        "positive": "very muscular, heavily defined muscles, athletic physique, visible definition",
        "negative": "soft untoned body, no visible muscle definition, sedentary physique",
        "scenes": "people_full_body", "hold_axes": (), "preserve": "scenes_and_objects",
    },
    "Expression (neutral <-> big smile)": {
        "positive": "beaming wide smile, joyful laughing expression, bright happy eyes",
        "negative": "flat neutral expression, unsmiling, blank affect, expressionless",
        "scenes": "people_portrait", "hold_axes": (), "preserve": "scenes_and_objects",
    },
    "Hair length (short <-> long)": {
        "positive": "very long flowing hair cascading well past the shoulders",
        "negative": "very short cropped hair, close-shorn, minimal hair length",
        "scenes": "people_portrait", "hold_axes": (), "preserve": "scenes_and_objects",
    },
    "Detail (smooth <-> highly detailed)": {
        "positive": "extremely detailed, intricate fine micro-texture, razor sharp, rich surface detail",
        "negative": "smooth flat simplified shapes, minimal texture, soft low-detail rendering",
        # "look" emits "a photograph" vs "an illustration", which carry detail level.
        "scenes": "mixed_general", "hold_axes": ("look",), "preserve": "bare_subjects",
    },
    "Lighting (dim <-> bright)": {
        "positive": "brightly lit, strong luminous key light, high-key exposure, radiant illumination",
        "negative": "dimly lit, deep shadow, low-key underexposed, murky darkness",
        # "context" emits "outdoors in daylight" / "against a dark backdrop".
        "scenes": "mixed_general", "hold_axes": ("context",), "preserve": "bare_subjects",
    },
    "Depth of field (deep <-> shallow bokeh)": {
        "positive": "shallow depth of field, creamy bokeh, strongly blurred background, subject isolated",
        "negative": "deep focus, everything sharp front to back, crisp detailed background",
        # "close-up" already implies shallow depth of field.
        "scenes": "mixed_general", "hold_axes": ("framing",), "preserve": "bare_subjects",
    },
    "Camera distance (close-up <-> wide shot)": {
        "positive": "wide establishing shot, subject small in a large environment, distant framing",
        "negative": "extreme close-up, subject filling the frame, tightly cropped macro framing",
        # "framing" IS this concept. Emitting it makes the base prompt contradict the phrase.
        "scenes": "mixed_general", "hold_axes": ("framing",), "preserve": "bare_subjects",
    },
    "Colour (muted <-> vivid)": {
        "positive": "vividly saturated, intense punchy colours, high chroma, bold colour contrast",
        "negative": "desaturated, muted washed-out palette, near-monochrome, faded colours",
        "scenes": "mixed_general", "hold_axes": ("look",), "preserve": "bare_subjects",
    },
    "Style (illustrated <-> photoreal)": {
        "positive": "photorealistic photograph, real camera optics, lifelike skin and materials",
        "negative": "stylised digital illustration, painterly rendering, obvious brushwork",
        # "look" IS this concept.
        "scenes": "mixed_general", "hold_axes": ("look",), "preserve": "bare_subjects",
    },
    "Weather (clear <-> stormy)": {
        "positive": "stormy overcast sky, heavy dark clouds, rain and wind, dramatic weather",
        "negative": "clear bright sky, calm still air, cloudless and settled weather",
        "scenes": "landscape", "hold_axes": ("context",), "preserve": "bare_subjects",
    },
    "Time of day (day <-> night)": {
        "positive": "at night, after dark, artificial lights against a black sky",
        "negative": "in broad daylight, bright midday sun, full daytime illumination",
        "scenes": "landscape", "hold_axes": ("context",), "preserve": "bare_subjects",
    },
    "Age of scene (new <-> weathered)": {
        "positive": "old and weathered, worn peeling surfaces, rust and patina, decayed with age",
        "negative": "brand new and pristine, spotless unworn surfaces, factory fresh",
        "scenes": "objects", "hold_axes": (), "preserve": "scenes_and_objects",
    },
    "Crowding (empty <-> crowded)": {
        "positive": "crowded and busy, packed with many people, dense bustling activity",
        "negative": "empty and deserted, no people at all, silent and vacant",
        "scenes": "architecture", "hold_axes": (), "preserve": "scenes_and_objects",
    },
}

DEFAULT_CONCEPT = "Age (young <-> old)"


def concept_spec(name):
    try:
        return CONCEPTS[name]
    except KeyError:
        raise ValueError("Unknown concept {!r}.".format(name))


# --------------------------------------------------------------------------------------
# scene generators
# --------------------------------------------------------------------------------------
# Composed as "{look} {framing} of {subject}, {context}", with held axes dropped entirely.

SUBJECT_PACKS = {
    "people_portrait": {
        "subject": ["a woman", "a man", "a person", "a teenager", "a shopkeeper",
                    "a musician", "a nurse", "a farmer", "a student", "a dancer",
                    "a soldier", "a chef"],
        "context": ["plain studio background", "on a busy street", "indoors by a window",
                    "in a garden", "against a brick wall", "in a dim cafe",
                    "outdoors on an overcast day", "in a library"],
        "framing": ["portrait", "headshot", "close-up portrait", "three-quarter portrait"],
        "look": ["a photograph", "a soft natural-light photo", "a studio photograph",
                 "a candid photo", "a digital painting"],
    },
    "people_full_body": {
        "subject": ["a woman", "a man", "a person", "an athlete", "a dancer",
                    "a construction worker", "a hiker", "a swimmer", "a cyclist", "a model"],
        "context": ["plain studio background", "in a gym", "on a beach",
                    "on a city sidewalk", "in a park", "in a locker room",
                    "against a white backdrop", "on a stage"],
        "framing": ["full body shot", "full length photo", "wide shot", "standing pose"],
        "look": ["a photograph", "a fashion photograph", "a fitness photo",
                 "a digital painting", "a candid photo"],
    },
    "landscape": {
        "subject": ["a mountain valley", "a pine forest", "a rocky coastline",
                    "a desert plain", "a wheat field", "a frozen lake", "a river gorge",
                    "rolling green hills", "a volcanic ridge", "a marshland"],
        "context": ["with distant hills", "with a winding path", "seen across water",
                    "with scattered trees", "under an open sky", "with low mist",
                    "with a small cabin", "with a dirt road"],
        "framing": ["wide landscape", "panoramic view", "scenic vista", "aerial view"],
        "look": ["a photograph", "a landscape photograph", "a matte painting",
                 "a digital painting", "a film still"],
    },
    "objects": {
        "subject": ["a leather armchair", "a brass telescope", "a ceramic teapot",
                    "a wooden guitar", "a bicycle", "a typewriter", "a stack of books",
                    "a pair of boots", "a wristwatch", "a bowl of fruit", "a toolbox",
                    "a suitcase"],
        "context": ["on a wooden table", "on a concrete floor", "against a plain backdrop",
                    "in a workshop", "on a windowsill", "in an attic", "on a shelf",
                    "on a market stall"],
        "framing": ["product shot", "still life", "close-up", "three-quarter view"],
        "look": ["a photograph", "a studio product photo", "a still life painting",
                 "a digital render", "a film photo"],
    },
    "architecture": {
        "subject": ["a train station concourse", "a narrow alley", "a cathedral interior",
                    "a shopping arcade", "a rooftop terrace", "an apartment courtyard",
                    "a subway platform", "a market hall", "a stone bridge", "a town square"],
        "context": ["in the early morning", "in the late afternoon", "under grey skies",
                    "with long shadows", "in the rain", "with neon signage",
                    "with parked bicycles", "with hanging lanterns"],
        "framing": ["wide interior shot", "street-level view", "symmetrical view",
                    "low angle shot"],
        "look": ["a photograph", "an architectural photograph", "a film still",
                 "a digital painting", "a documentary photo"],
    },
    "animals": {
        "subject": ["a red fox", "a tabby cat", "a golden retriever", "a barn owl",
                    "a grey wolf", "a horse", "a rabbit", "a raven", "a deer", "a tiger"],
        "context": ["in a snowy forest", "on a grassy field", "on a rocky outcrop",
                    "beside a river", "in tall grass", "against a plain backdrop",
                    "in a barn", "at the treeline"],
        "framing": ["close-up", "full body shot", "portrait", "wide shot"],
        "look": ["a photograph", "a wildlife photograph", "a digital painting",
                 "an oil painting", "a nature documentary still"],
    },
    "anime_illustration": {
        "subject": ["a young woman", "a young man", "a schoolgirl", "a swordsman",
                    "a witch", "a knight", "a magical girl", "a pilot", "a shrine maiden",
                    "a street musician"],
        "context": ["on a rooftop at sunset", "in a classroom", "in a snowy street",
                    "in a bamboo forest", "at a summer festival", "in a neon-lit alley",
                    "on a train platform", "in a flower field"],
        "framing": ["portrait", "upper body shot", "full body illustration", "dynamic pose"],
        "look": ["an anime illustration", "a manga-style drawing", "a cel-shaded artwork",
                 "a soft anime painting", "a detailed anime key visual"],
    },
    "mixed_general": {
        "subject": ["a woman", "a man", "a red fox", "a mountain valley",
                    "a leather armchair", "a train station concourse", "a bowl of fruit",
                    "a city skyline", "a young woman in a coat", "a pine forest",
                    "a bicycle", "a barn owl"],
        "context": ["on a plain background", "outdoors in daylight", "indoors by a window",
                    "on a city street", "in a quiet room", "under an open sky",
                    "in the late afternoon", "against a dark backdrop"],
        "framing": ["close-up", "medium shot", "wide shot", "portrait"],
        "look": ["a photograph", "a digital painting", "a film still", "an illustration",
                 "a studio photo"],
    },
}

DEFAULT_SUBJECT_PACK = "mixed_general"
_AXES = ("look", "framing", "subject", "context")


def _compose(look, framing, subject, context):
    """Join the four axes, skipping any that were held out."""
    head = " ".join(p for p in (look, framing) if p)
    lead = "{} of {}".format(head, subject) if head else subject
    return "{}, {}".format(lead, context) if context else lead


def expand_base_prompts(pack_name, count, seed=0, hold_axes=()):
    """Deterministically draw ``count`` varied scenes, omitting ``hold_axes`` entirely.

    Holding an axis is not cosmetic. A scene that already says "wide shot" cannot be used
    to measure a wide-shot concept: the base prompt and the concept phrase describe the
    same attribute, so the measured direction is contradiction rather than concept.

    The full cross product is enumerated and shuffled with a fixed seed rather than
    sampled, so the set is duplicate-free and a rebuild at the same seed is identical.
    """
    if pack_name not in SUBJECT_PACKS:
        raise ValueError("Unknown scene pack {!r}. Available: {}.".format(
            pack_name, ", ".join(sorted(SUBJECT_PACKS))))
    axes = SUBJECT_PACKS[pack_name]
    hold = set(hold_axes or ())
    if "subject" in hold:
        raise ValueError("The 'subject' axis cannot be held -- a scene needs a subject.")

    values = {a: ([""] if a in hold else axes[a]) for a in _AXES}
    combos = [
        _compose(look, framing, subject, context)
        for subject in values["subject"]
        for context in values["context"]
        for framing in values["framing"]
        for look in values["look"]
    ]
    combos = list(dict.fromkeys(combos))
    random.Random(seed).shuffle(combos)
    return combos[: max(1, int(count))]


def build_triplets(base_prompts, positive_phrase, negative_phrase):
    """[(neutral, positive, negative)] -- the three prompts per scene.

    The extremes are the neutral prompt with a phrase appended, so the three share an
    identical prefix. The solver relies on that: it compares the three only over the
    shared prefix positions, which is the only region a LoRA on the neutral prompt can
    actually affect.
    """
    triplets = []
    for base in base_prompts:
        stem = base.strip().rstrip(".")
        triplets.append((
            "{}.".format(stem),
            "{}. {}.".format(stem, positive_phrase.strip().rstrip(".")),
            "{}. {}.".format(stem, negative_phrase.strip().rstrip(".")),
        ))
    return triplets


# --------------------------------------------------------------------------------------
# preservation sets
# --------------------------------------------------------------------------------------
# "scenes_and_objects" complements subject-attribute sliders (age, weight, expression):
#     rich scenes with no people whose attributes could be dragged along.
# "bare_subjects" complements global-look sliders (detail, camera, colour, lighting):
#     plain subject nouns with no framing, lighting, style or time vocabulary at all, so
#     what is preserved is *what is in the picture*, not how it is photographed.

PRESERVATION_PACKS = {
    "scenes_and_objects": [
        "a busy market street at noon",
        "a sailing ship in a storm",
        "a bowl of ripe strawberries on a wooden table",
        "a snow-covered mountain range",
        "a vintage car parked outside a diner",
        "a stack of old leather-bound books",
        "a cat asleep on a windowsill",
        "a modern glass office tower",
        "a plate of pasta with fresh basil",
        "a wooden rowboat on a still lake",
        "a field of sunflowers",
        "an astronaut floating above the earth",
        "a cobblestone street in an old european town",
        "dew on a spider web",
        "a bustling train station",
        "a lighthouse on a rocky cliff",
        "a bicycle leaning against a brick wall",
        "a bowl of steaming ramen with an egg",
        "a desert canyon",
        "an antique brass pocket watch",
        "a forest path covered in autumn leaves",
        "a hot air balloon over rolling farmland",
        "a violin resting on sheet music",
        "a herd of elephants crossing a savannah",
        "a bakery window full of pastries",
        "a fishing village harbour",
        "a chess board mid-game",
        "a waterfall in a tropical jungle",
        "a chalkboard covered in equations",
        "a greenhouse full of ferns",
    ],
    "bare_subjects": [
        "a wooden chair", "a golden retriever", "a bowl of apples", "a brick house",
        "a violin", "a bicycle", "a red fox", "a coffee cup", "a pine tree",
        "a stone bridge", "a pair of leather boots", "a barn owl", "a teapot",
        "a mountain", "a sailing boat", "a horse", "a bookshelf", "a guitar",
        "a lighthouse", "a bunch of grapes", "a wristwatch", "a tabby cat",
        "a wheat field", "a suitcase", "a church", "a raven", "a typewriter",
        "a river", "a bouquet of roses", "a wolf", "a telescope", "a windmill",
    ],
    "people_and_faces": [
        "a smiling woman in a red coat",
        "a man reading a newspaper in a cafe",
        "a group of friends laughing around a table",
        "a chef plating food in a busy kitchen",
        "a violinist performing on a stage",
        "a construction worker in a hard hat",
        "two people dancing at a wedding",
        "a doctor in scrubs in a hospital corridor",
        "a street vendor selling flowers",
        "a runner crossing a finish line",
        "a grandmother knitting by a fireplace",
        "a barista pouring latte art",
        "a teacher writing on a whiteboard",
        "a fisherman mending nets on a dock",
        "a painter at an easel in a studio",
        "a librarian shelving books",
    ],
    "style_and_medium": [
        "a watercolour painting of a village square",
        "a charcoal sketch of a human hand",
        "a 1970s film photograph with visible grain",
        "a flat vector illustration of a city skyline",
        "a baroque oil painting with dramatic chiaroscuro",
        "a pencil architectural drawing with clean linework",
        "a japanese woodblock print of a wave",
        "a low-poly 3d render of a landscape",
        "an art nouveau poster with ornate borders",
        "a black and white documentary photograph",
        "a pixel art scene of a forest",
        "a stained glass window design",
        "an impressionist painting of a garden",
        "a technical blueprint of a machine",
    ],
}

DEFAULT_PRESERVATION_PACK = "scenes_and_objects"


def expand_preservation(pack_name, count, seed=0):
    """Deterministically draw ``count`` preservation prompts, or [] for 'off'."""
    if pack_name in ("off", "none", ""):
        return []
    if pack_name not in PRESERVATION_PACKS:
        raise ValueError("Unknown preservation pack {!r}. Available: {}.".format(
            pack_name, ", ".join(sorted(PRESERVATION_PACKS)) + ", off"))
    prompts = list(PRESERVATION_PACKS[pack_name])
    random.Random(seed).shuffle(prompts)
    return prompts[: max(0, int(count))]


def split_lines(text):
    """Multiline widget text -> list of non-empty, non-comment lines."""
    return [ln.strip() for ln in (text or "").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
