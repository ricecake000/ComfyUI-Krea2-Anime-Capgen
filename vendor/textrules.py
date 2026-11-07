"""Shared text rules for krea2-capgen: phrase segmentation, facet rules, group assignment, tokenization for
tag realization (lexical matching with synonym classes). Pure standard library; imported by build_model.py and
capgen.py (eval.py deliberately re-implements its own checker)."""
import re

# ---------------------------------------------------------------- groups / facets
APPEAR, OUTFIT, ACTION, PLACE, TAIL, SUBJ = 0, 1, 2, 3, 4, 5
GROUP_NAMES = ['appearance', 'outfit', 'action', 'place', 'tail', 'subject']
NG = 5  # APPEAR, OUTFIT, ACTION, PLACE, TAIL carry donor phrases; SUBJ is the subject slot
SCENE_GROUPS = (ACTION, PLACE)

FACETS = ['subject', 'style', 'shot', 'light', 'eyes', 'hair', 'hairacc', 'expr', 'body', 'exposure', 'pose',
          'outfit', 'setting', 'weakstyle', 'eyestate', 'strongstyle']
FBIT = {f: 1 << i for i, f in enumerate(FACETS)}
# ordering facet (one per phrase) - precedence for the "primary" facet used for positions and trimming
PRIMARY_ORDER = ['subject', 'style', 'shot', 'hair', 'eyes', 'eyestate', 'expr', 'pose', 'light', 'setting',
                 'body', 'exposure', 'hairacc', 'outfit', 'weakstyle', 'misc']
PRIMARY = {f: i for i, f in enumerate(PRIMARY_ORDER)}

COLOR_WORDS = ('white|black|grey|gray|silver|silvery|blonde|blond|golden|gold|yellow|red|crimson|scarlet|pink|'
               'magenta|fuchsia|purple|violet|lavender|lilac|mauve|blue|azure|navy|cyan|aqua|teal|turquoise|green|'
               'emerald|mint|brown|chestnut|auburn|orange|amber|beige|cream|platinum|ash|ashen|rose|peach|'
               'multicolored|multicolor|rainbow|two-tone|maroon|burgundy|wine|copper|bronze|tan|ivory|sapphire|'
               'ruby|indigo|olive|khaki|honey|caramel|strawberry|periwinkle|cerulean|cobalt|jade|coral|salmon|'
               'charcoal|ebony|raven|jet-black|snow-white|pastel|bright|dark|light|pale|deep|vivid|vibrant|soft|'
               'blonde-ish|reddish|pinkish|bluish|greenish|purplish|whitish|greyish|grayish')
_PURE_COLORS = set(COLOR_WORDS.split('|')) - {'pastel', 'bright', 'dark', 'light', 'pale', 'deep', 'vivid',
                                               'vibrant', 'soft'}
_SHADE = r'(light|dark|bright|pale|deep|soft|pastel|vivid|vibrant|dusty|muted|hot|baby|sky|neon)'
_SHADE_BEFORE_COLOR = re.compile(r'\b' + _SHADE + r'[- ](?=(' + COLOR_WORDS + r')\b)')

_R = {
    'subject': r'\billustration\b|^(an? |one |the )?(anime |chibi |cute |young |beautiful )*(girl|woman|lady)$',
    'style': (r'\b(shading|shaded|shadings|cel|cel-shaded|line ?art|lineart|linework|line work|lines|outlines?|'
              r'painterly|rendering|rendered|palette|digital (art|painting|illustration)|coloring|colouring|'
              r'art style|anime style|illustration style|drawing style|style|aesthetic|brush ?strokes?|'
              r'watercolou?r|sketch(y)?|airbrush\w*|gradients?|halftone|screentone|texture[sd]?|'
              r'colou?r scheme|tones?|hues|saturat\w*|contrast|detailed|high detail|highly detailed|'
              r'colou?rs|colou?red|monochrome|greyscale|grayscale|sepia|flat colou?rs?|limited colou?r)\b'),
    'weakstyle': (r"\b(signature|signed|watermark|text|texts|logo|logos|artist name|username|copyright|caption|"
                  r"captions|credit|credits|letters|lettering|written|writing|reads|reading|says|saying|"
                  r"handle|url|twitter|patreon|pixiv|kanji|japanese characters|characters|numbers?|"
                  r"speech bubbles?|dialogue|sound effects?|onomatopoeia|title|stamp|date)\b"),
    'shot': (r'\b(shot|close-?up|closeup|portrait|angle|view|viewpoint|framing|framed|perspective|composition|'
             r'pov|first-person|point of view|from (above|below|behind|the side|side|the front|front)|'
             r'upper body|full body|full-body|cowboy|dutch|foreshortening|fisheye|wide-angle|headshot|cropped|'
             r'centered|off-center|zoomed|overhead|bird\'s-eye|worm\'s-eye|three-quarter|profile|'
             r'facing (the )?(viewer|camera|away|left|right|forward|sideways)|depth of field|blurred foreground|'
             r'foreground|lower body|waist up|thighs up|knees up|mid-thigh|bust)\b'),
    'light': (r'\b(light(ing|s)?|lit|glow(ing|s|y)?|sunlight|sunlit|moonlight|moonlit|shadows?|shade|backlit|'
              r'backlighting|backlight|illuminat\w*|sunset|sunrise|dusk|dawn|twilight|golden hour|bokeh|'
              r'lens flare|daylight|sunny|sunbeams?|light rays|rays|rim light|neon|spotlights?|dim|'
              r'dimly|darkness|atmosphere|atmospheric|ambient|haze|hazy|sparkles?|sparkling|sparkly|'
              r'glitter\w*|shimmer\w*|glint\w*|reflections?|reflective|luminous|radiant|'
              r'cinematic|dramatic|moody|overcast|chromatic aberration|film grain|bloom|vignette|light leaks?|'
              r'highlighting|gleam\w*|shine|shiny|shining|glare)\b'),
    'eyes': (r'\beyes?\b|\bpupils?\b|heterochromia|eyelashes|lashes|irises|\biris\b|eyeliner|eyeshadow|'
             r'eye makeup|\beyebrows?\b|\bbrows\b'),
    'hair': (r'\bhair\b|\bbangs\b|ponytails?|twintails|twin tails|pigtails|braids?|braided|ahoge|'
             r'\bbuns?\b|sidelocks|hime cut|bob cut|\bbob\b|hairstyles?|haircut|\blocks\b|ringlets|drills|'
             r'tresses|inner hair|ombre|odango|'
             r'drill hair|side ponytail|hair intakes|bob-cut|\bmane\b|twin buns|side tails?|twin drills|'
             r'pixie cut|\bbraid\b|haired|antenna hair|\bplaits?\b'),
    'hairacc': (r'\bhair ?(ribbons?|ornaments?|bows?|flowers?|clips?|pins?|bands?|ties?|accessor\w*|'
                r'sticks?|tubes?|rings?|beads|bobbles?|scrunchies?|cuffs?)\b|hairclips?|hairpins?|hairbands?|'
                r'scrunchies?|kanzashi|barrettes?|hair-ornament'),
    'expr': (r'\b(smil(e|es|ing)|blush(es|ing|ed)?|expression|expressions|grin(s|ning)?|frown(s|ing)?|tears?|'
             r'teary|tearful|crying|cries|sobbing|pout(s|ing|y)?|open mouth|open-mouthed|closed mouth|'
             r'mouth|lips|looking|looks|gaze|gazing|glanc\w*|stare|staring|wink(s|ing)?|fangs?|teeth|tongue|'
             r'embarrassed|surprised|shocked|startled|angry|annoyed|irritated|calm|serene|happy|cheerful|'
             r'joyful|sad|sorrowful|shy|playful|confident|gentle|sleepy|drowsy|tired|smug|serious|determined|'
             r'focused|neutral|blank|bored|nervous|flustered|excited|content|relaxed|dreamy|melancholic|'
             r'melancholy|wistful|pensive|mischievous|seductive|sultry|teasing|kind|warm smile|lipstick|drool\w*|'
             r'sweat(ing|y)?|sweatdrops?|sweat drops?|laugh(s|ing)?|giggl\w*|yawn\w*|eyes closed|closed eyes|'
             r'half-closed|squint\w*|furrowed|frowning|scowl\w*|glare|glaring|sticking out|blep|:3|'
             r'speechless|curious|thoughtful|innocent|lonely|longing|loving|affectionate|tender|soft gaze|'
             r'emotion\w*|mood|face|facial|cheeks?|flushed|panting|breath(ing)?|heavy breathing|steam|'
             r'expressionless|deadpan|stoic|cold|fierce|intense|menacing|evil|crazy|yandere|ahegao|smirk\w*|'
             r'excitement|happiness|joy|delight|wonder|awe|fear|scared|afraid|worried|anxious|upset|'
             r'pained|hurt|in pain|mouth open|lip bite|biting (her|his)? ?lip)\b'),
    'body': (r'\b(ears?|eared|tails?|horns?|horned|wings?|winged|halo|breasts?|bust|chest|skin|skinned|thighs?|'
             r'freckles|mole|moles|beauty mark|elf|elven|animal|fangs|tattoos?|scars?|scarred|nails|'
             r'nail polish|manicure|fingernails|muscles?|muscular|abs|curvy|petite|slender|slim|figure|'
             r'physique|build|tan|tanned|tanlines|tan lines|fur|furry|paws?|claws?|scales|antennae|antlers|'
             r'kemonomimi|body|bodied|hips|waist|legs|long legs|plump|chubby|toned|athletic|height|tall|'
             r'short stature|small frame|flat chest|cleavage|makeup|lipstick|blush stickers|markings|'
             r'facial mark|whisker\w*|stigmata|third eye|heterochromatic|mechanical (arm|parts|limbs?)|'
             r'prosthetic|robotic|android|cyborg|joints|doll joints|pointy|pointed|kitsune|demon|devil|angel|'
             r'succubus|vampire|ghost|slime|monster girl|mermaid|centaur|lamia|harpy|insect|feathered|'
             r'heart-shaped tail|spade-tipped|soles|toes|belly|butt)\b'),
    'exposure': (r'\b(bare (shoulders|arms|legs|feet|back|chest|thighs|midriff|skin|neck|hands)|barefoot|'
                 r'barefooted|cleavage|midriff|navel|underboob|sideboob|exposed|zettai ryouiki|'
                 r'collarbones?|nude|naked|topless|bottomless|nipples?|pubic|thong|revealing|skimpy|'
                 r'groin|crotch|cameltoe|pantyshot|panty shot|upskirt|downblouse|shirt lift|lifted|'
                 r'see-through|wardrobe malfunction|areola\w*)\b'),
    'pose': (r'\b(sitting|sits|seated|standing|stands|lying|lies|laying|kneeling|kneels|leaning|leans|walking|'
             r'walks|running|runs|jumping|jumps|squatting|squats|crouching|crouched|crouches|holding|holds|held|'
             r'hugging|hugs|reaching|reaches|raising|raises|raised|resting|rests|pose|poses|posing|posed|'
             r'fist|fists|peace sign|v sign|salute|saluting|stretching|stretches|floating|floats|dancing|dances|'
             r'sleeping|sleeps|asleep|eating|eats|drinking|drinks|playing|plays|looking back|turning|turns|'
             r'turned|bending|bends|bent|crossed|clasped|clasping|interlaced|covering|covers|touching|touches|'
             r'grabbing|grabs|gripping|grips|carrying|carries|lifting|lifts|pulling|pulls|adjusting|adjusts|'
             r'tugging|tying|brushing|wiping|biting|bites|licking|licks|sipping|sips|drawing|'
             r'singing|sings|typing|cooking|swimming|swims|wading|bathing|showering|gesture|gestures|gesturing|'
             r'heart hands|thumbs up|outstretched|extended|spread|tilted|tilting|tilts|tilt|head tilt|'
             r'cocked|perched|sprawled|reclining|reclines|lounging|lounges|arching|arched|twisting|spinning|'
             r'waving|waves at|pointing|points|cradling|clutching|clutches|cupping|embracing|straddling|'
             r'riding|rides|flying|flies|falling|falls|mid-air|midair|fighting|swinging|wielding|wields|'
             r'aiming|aims|casting|attacking|blowing|kissing|peeking|peeks|hiding|hides|waiting|posture|'
             r'stance|prone|supine|on all fours|all fours|on her back|on her side|on her stomach|'
             r'over (her|the|one) shoulder|behind her back|behind her head|hands? on|arms? behind|'
             r'one knee|knee up|knees up|legs together|leg up|hugging knees|seiza|wariza|indian style|'
             r'cross-legged|curled up|kick\w*|punch\w*|step\w*|stride|striding|leap\w*|lean|sit|stand|'
             r'kneel|squat|crouch|lie|hold|offering|offers|presenting|presents|handing|'
             r'feeding|petting|pets|playing with|using|uses|taking|takes|selfie|phone|smartphone|'
             r'pressed|pressing|presses|propped|supporting|supported|balanced|balancing|dangling|dangles|'
             r'swaying|sways|motion|moving|'
             r'action|dynamic|mid-motion|mid-stride|mid-jump|mid-step)\b'),
    'outfit': (r'\b(wearing|wears|worn|dressed|clad|dress|dresses|shirt|shirts|skirt|skirts|jacket|jackets|coat|'
               r'coats|uniform|uniforms|sweater|sweaters|hoodie|hoodies|bikini|bikinis|swimsuit|swimsuits|'
               r'kimono|kimonos|apron|aprons|gloves?|socks?|thighhighs|thigh-highs|thigh highs|pantyhose|'
               r'stockings?|boots?|shoes?|sneakers|loafers|heels|high heels|hat|hats|cap|caps|ribbons?|bows?|'
               r'bowtie|choker|chokers|necklace|necklaces|earrings?|jewelry|jewellery|jewels?|bracelets?|'
               r'belts?|tie|ties|necktie|neckerchief|scarf|scarves|cape|capes|cloak|cloaks|armor|armour|'
               r'armored|leotard|shorts|pants|trousers|jeans|bra|bras|panties|underwear|lingerie|vest|vests|'
               r'blouse|blouses|collar|collared|sleeves?|sleeved|sleeveless|headband|headdress|headphones|'
               r'glasses|eyeglasses|spectacles|mask|masks|crown|crowns|tiara|hairband|bodysuit|outfit|outfits|'
               r'costume|costumes|attire|clothes|clothing|garment|garments|top|tops|hood|hooded|robe|robes|'
               r'corset|garter|garters|frills?|frilled|frilly|lace|laced|lacy|maid|bodice|bustier|gown|gowns|'
               r'sundress|frock|tunic|camisole|cami|tank top|crop top|tube top|halter|halterneck|turtleneck|'
               r'cardigan|blazer|parka|windbreaker|tracksuit|jersey|pajamas|pyjamas|nightgown|negligee|'
               r'chemise|yukata|hakama|cheongsam|qipao|china dress|hanfu|sarashi|fundoshi|serafuku|sailor|'
               r'miniskirt|pleated|ruffled|ruffles|stockings|leggings|tights|kneehighs|kneesocks|legwear|'
               r'footwear|sandals|slippers|geta|zori|mary janes|pumps|stilettos|gauntlets|bracers|armlets|'
               r'arm warmers|leg warmers|wristbands?|wrist cuffs|cuffs|anklets?|rings?|pendants?|brooch|'
               r'amulet|charms?|beads|piercings?|beret|bonnet|helmet|visor|beanie|fedora|witch hat|veil|'
               r'goggles|eyewear|sunglasses|monocle|eyepatch|bandages?|bandaid|band-aid|bandana|sash|obi|'
               r'lanyard|badges?|emblem|insignia|patch|patches|straps?|strapless|strappy|harness|holster|'
               r'backpack|bag|bags|purse|handbag|satchel|pouch|zipper|zip|buttons?|buttoned|unbuttoned|'
               r'pockets?|hem|hemline|trim|trimmed|neckline|off-shoulder|off-the-shoulder|backless|sheer|'
               r'mesh|fishnets?|latex|leather|denim|silk|silky|satin|velvet|fur-trimmed|knit|knitted|plaid|'
               r'striped|stripes|polka|checkered|camouflage|towel|bandeau|sports bra|bloomers|buruma|overalls|'
               r'suspenders|mantle|poncho|shawl|stole|muffler|earmuffs|mittens|fingerless|thigh strap|'
               r'leg garter|leash|circlet|diadem|headpiece|headgear|headwear|headset|earpiece|ornaments?|'
               r'ornamented|accessor(y|ies)|gems?|gemstones?|cufflinks|epaulettes|epaulets|tassels?|'
               r'fringed|pom-poms?|pompoms?|frill|layered|oversized|baggy|loose-fitting|tight-fitting|'
               r'form-fitting|fitted|cropped top|crop|tank|tee|t-shirt|polo|overcoat|trench coat|raincoat|'
               r'peacoat|kimono-style|bathrobe|apron dress|pinafore|jumpsuit|romper|dungarees|catsuit|'
               r'plugsuit|wetsuit|one-piece|two-piece|school swimsuit|competition swimsuit|race queen|'
               r'nurse|police|military|sailor collar|business suit|suit|tuxedo|waistcoat|cravat|ascot|'
               r'jabot|bolero|shrug|capelet|kimono sleeves|detached sleeves|wide sleeves|puffy sleeves|'
               r'juliet sleeves|bell sleeves|cuff|hairpiece|fascinator|flower crown|wreath|garland|lei|'
               r'name tag|nametag|id card|wristwatch|watch|scrunchie|bangle|bangles|anklet|toe ring|'
               r'arm band|armband|armbands|leg band|thigh band|thighband|gartered|wrapped|wrap|wraps|'
               r'shoulder pads|pauldrons?|breastplate|chainmail|greaves|vambraces|tabard|surcoat|habit|'
               r'wimple|coif|cassock|vestments|frills|petticoat|crinoline|bloomers|drawers|briefs|shoe|'
               r'stiletto|platform|wedges|clogs|moccasins|uwabaki|socks|loose socks|ankle socks|'
               r'over-the-knee|overknee|knee-high|thigh-high|calf-length|ankle-length|floor-length|'
               r'knee-length|mini|midi|maxi|long-sleeved|short-sleeved|puff-sleeved|frock coat|lab coat|'
               r'labcoat|scrubs|gakuran|blazer jacket|cardigan sweater|sweater vest|chiton|toga|sarong|'
               r'pareo|kimono jacket|haori|happi|jinbei|dougi|hakama skirt|tabi|dogi|chains?|chained|floral|patterned|'
               r'print|printed|motif|embroidered|embroidery|crystal|crystals|metallic|cross-laced|lace-up)\b'),
    'setting': (r'\b(background|backgrounds|backdrop|sky|skies|clouds?|cloudy|outdoors?|outdoor|outside|indoors?|'
                r'inside|interior|room|rooms|street|streets|city|cities|cityscape|skyline|town|village|'
                r'forest|woods|woodland|jungle|beach|beaches|ocean|sea|seaside|shore|shoreline|coast|water|'
                r'waters|field|fields|meadow|garden|gardens|flowers?|blossoms?|petals?|trees?|branches|'
                r'leaves|leaf|foliage|bushes|plants?|vines|moss|grass|grassy|window|windows|windowsill|'
                r'classroom|cafe|café|coffee shop|bedroom|bed|beds|bath|bathroom|bathtub|onsen|hot spring|'
                r'pool|poolside|night|nighttime|evening|morning|afternoon|midnight|daytime|day|snow|snowy|'
                r'snowing|snowflakes|rain|rainy|raining|raindrops|rainfall|stars?|starry|moon|moons|sun|'
                r'sand|sandy|wall|walls|floor|floors|ground|building|buildings|architecture|scenery|scene|'
                r'scenic|setting|landscape|stage|park|river|lake|pond|mountains?|hills?|valley|cliffs?|'
                r'cave|desert|island|table|tables|desk|chair|chairs|bench|sofa|couch|shrine|temple|church|'
                r'cathedral|castle|palace|tower|house|home|apartment|kitchen|library|train|station|bridge|'
                r'ruins|space|outer space|underwater|waves|wave|surf|tide|horizon|sunshine|weather|storm|'
                r'lightning|thunder|aurora|galaxy|nebula|cosmos|planet|universe|void|abyss|grid|'
                r'wallpaper|white background|simple background|plain|abstract|geometric|confetti|'
                r'balloons?|lanterns?|lamps?|candles?|decorations?|decorated|furniture|shelf|shelves|'
                r'bookshelf|bookshelves|books?|curtains?|carpet|rug|tatami|futon|pillows?|cushions?|blanket|'
                r'blankets|sheets|bedsheets?|mattress|counter|car|cars|vehicle|bicycle|bike|motorcycle|boat|'
                r'ship|airplane|platform|tracks|railway|road|path|pathway|sidewalk|crosswalk|signs?|signboard|'
                r'billboard|poles?|wires|power lines|utility pole|fence|railing|railings|gate|torii|pillars?|'
                r'columns?|arch|archway|doorway|door|doors|ceiling|tiles|tiled|mirror|aquarium|fish|birds?|'
                r'butterfl(y|ies)|feathers|bubbles|droplets|splash\w*|splashing|ice|icicles|frost|frosty|fog|'
                r'foggy|mist|misty|smoke|smoky|fire|flames?|embers|sparks|lava|steam|puddles?|waterfall|'
                r'fountain|rocks?|rocky|stones?|pebbles|cherry blossoms?|sakura|bamboo|autumn|fall foliage|'
                r'winter|summer|spring|season\w*|festival|fireworks|shop|store|market|restaurant|bar|office|'
                r'school (building|hallway|corridor|rooftop|grounds|gate|courtyard)|schoolyard|hallway|corridor|staircase|stairs|steps|alley|alleyway|rooftop|roof|balcony|'
                r'terrace|porch|veranda|courtyard|plaza|square|fountain|gym|stadium|arena|locker room|'
                r'hospital|laboratory|lab|spaceship|cockpit|throne|altar|dungeon|battlefield|warzone|'
                r'cityscape|neon signs|vending machine|convenience store|bus stop|subway|street lamp|'
                r'streetlight|lamppost|lantern|paper lanterns|wind chime|surroundings|scattered|'
                r'environment|location|landmark|tables|picnic|blanket|food|drinks?|cup|cups|mug|teacup|'
                r'teapot|plate|plates|bowl|bowls|tray|cake|cakes|dessert|desserts|sweets|candy|fruit|fruits|'
                r'bottle|bottles|glass of|wine glass|can|cans|snacks|bread|ice cream|parfait|coffee|tea|'
                r'umbrella|parasol|weapon|weapons|sword|swords|katana|blade|gun|guns|rifle|pistol|staff|'
                r'wand|spear|bow and arrow|shield|scythe|axe|book|guitar|microphone|instrument|camera|'
                r'flag|banner|sign|cards?|letter|envelope|gift|present|bouquet|box|boxes|basket|baskets|'
                r'toy|toys|plush\w*|stuffed (animal|toy)|teddy bear|doll|dolls|mascot|mascots|pet|cat|cats|'
                r'kitten|kittens|dog|dogs|puppy|bunny|rabbit|bird|creature|creatures|dragon|dragons|'
                r'chess|game|controller|console|computer|laptop|monitor|screen|keyboard|television|tv|'
                r'speakers?|cable|cables|rope|ropes|cage|orbs?|'
                r'magic circle|magic|magical|spell|energy|aura|particles|effects?|motion lines|speed lines|'
                r'hearts?|musical notes|notes|symbols?|emoji|stickers?|floating|drifting|scattered around|'
                r'surrounded|surrounding|around her|behind her|beside her|next to her|in front of her|'
                r'on the left|on the right|in the background|in the foreground|left side|right side|'
                r'frame|border|panel|panels|vignette)\b'),
}
# Facet order matters only for _R construction; matching is independent per facet.
FACET_RX = {f: re.compile(rx) for f, rx in _R.items()}
_PLAIN_ANIMAL_EARS = re.compile(r'\b(cat|fox|dog|wolf|rabbit|bunny|bear|mouse|raccoon|horse|cow|sheep|'
                                r'tiger|lion|deer|squirrel|bat|monkey|leopard|goat|pig|tanuki|lop|dragon|demon|'
                                r'devil|ram|oni|angel|fairy|butterfly|bird|feathered|insect|fish|shark|snake|'
                                r'kitsune|succubus|cat-like|fox-like|animal)[- ](ears?|eared|tails?|girl|paws?|'
                                r'horns?|wings?|scales|fins?|fangs|ear)\b')
_LIMBS = re.compile(r'\b(hands?|arms?|fingers?|fingertips|legs?|feet|foot|knees?|palms?|thumbs?|elbows?|'
                    r'wrists?|shoulders?|chin|cheek|head)\b')
_DECOR = re.compile(r'\b(flowers?|star|stars|hearts?|moon|crescent|butterfl(y|ies)|petals?|feathers?|'
                    r'snowflakes?|cherry blossoms?|sakura|leaf|leaves|bells?|cross|skull|clouds?|fish|'
                    r'bird|birds|cat|cats|bunny|rabbit|dragon|sun|rose|roses|lily|lilies|frog|bear|shell|shells|'
                    r'sea|ocean|wave|waves|water|ice|fire|flames?|lightning|spider|web|gem|orbs?|candy|food|'
                    r'fruit|strawberr(y|ies)|cherr(y|ies)|bow|ribbon|lace|stones?|pattern|motifs?|book|'
                    r'sword|swords|weapon|key|keys|card|cards|carrot|plush|toy|heart-shaped|star-shaped)\b')
_OUTFIT_COMPOUND = re.compile(
    r'\b(gym|school|beach|swim|bath|night|summer|winter|spring|party|office|rain|snow|sun|kitchen|bed|house|'
    r'room|shrine|church|tea|garden|ocean|sea|star|moon|flower|sky|cloud|water|fire|ice|space|military|'
    r'street|city|forest|desert|festival|christmas|halloween|stage|idol|race|track|tennis|baseball|'
    r'basketball|volleyball|soccer|track|sports|ski|snowboard|ballet|dance)\s+(shorts|uniforms?|swimsuits?|'
    r'dress|dresses|towels?|hats?|coats?|jackets?|shirts?|clothes|wear|gowns?|boots|shoes|socks|skirts?|'
    r'robes?|aprons?|bags?|caps?|visors?|kimonos?|outfits?|attire|costumes?|slippers|sandals|vest|'
    r'leotards?|bloomers|buruma|pants|trousers|sweater|cardigan|blazer|tie|ribbon)\b')
_STRONG_STYLE = re.compile(r'\b(shading|shaded|cel|line ?art|lineart|linework|lines|palette|rendering|painterly|'
                           r'coloring|colouring|art style|anime style|brush ?strokes?|watercolou?r|digital)\b')
_LIMB_LEAD = re.compile(r'^(one |both |her |left |right )*(hands?|arms?|fingers?|legs?|feet|knees?|palms?)\b')
_SHINY = re.compile(r'\b(shiny|sparkling|sparkly|glowing|glittering|glittery|shimmering|shimmery|reflective|'
                    r'luminous|radiant|gleaming|glossy|metallic|lit|neon|glow)\b')
_HAIR2 = re.compile(r'\b(' + COLOR_WORDS + r')\b.*\b(streaks?|highlights|tips|ends|strands?|locks|inner|'
                    r'underside|gradient|curls)\b')
_HAIR_LOC = re.compile(r'\b(in|on|through|into|from|behind|over|across|against|with) (her|the|its) hair\b|'
                       r'\bhair (ribbon|bow|ornament|clip|pin|flower)s?\b')
_EYE_APPEAR = re.compile(r'\b(' + COLOR_WORDS + r')\b|\bpupils?\b|heterochromia|lashes|eyelashes|irises|\biris\b|'
                         r'slit|heart-shaped|star-shaped|symbol|sparkling eyes|glowing eyes|eyeliner|eyeshadow|'
                         r'tsurime|tareme|jitome|almond-shaped|large eyes|big eyes|eyes with|-eyed\b|eye color|'
                         r'eye colour|colored eyes|coloured eyes')
_EYE_STATE = re.compile(r'\b(closed|closing|half-closed|half closed|wink\w*|squint\w*|shut|one eye|teary|'
                        r'tearful|narrowed|looking|wide-eyed|wide eyes|teardrop|tears|crying|rolling|averted|'
                        r'downcast|upturned|eyes closed)\b')
_HAIR_WORD_ONLY_ACC = re.compile(r'\bhair ?(ribbons?|ornaments?|bows?|flowers?|clips?|pins?|bands?|ties?|'
                                 r'accessor\w*|sticks?|tubes?|rings?|beads|bobbles?|scrunchies?|cuffs?)\b|'
                                 r'hair-ornament')
_SUBJ_NP = re.compile(
    r'^((a|an|one|the)\s+)?([\w\'-]+\s+){0,3}?(girl|boy|woman|man|lady|maiden|catgirl|foxgirl|dog girl|cat girl|'
    r'fox girl|wolf girl|schoolgirl|heroine|princess|witch|knight|warrior|nun|idol|student|vampire|'
    r'female|figure|person|character|youngster|kid|child|teenager|teen|gal|beauty|maid|bunny girl|'
    r'demon girl|elf|android|succubus|mermaid|kunoichi|ninja|priestess|shrine maiden|miko)s?'
    r'(\s+(with|in|wearing|sitting|standing|holding|lying|looking|on|at|from|who|and|posing|kneeling|'
    r'leaning|walking|is|dressed|clad|having|seen|shown|depicted|against|under|by|near|inside|outdoors|'
    r'indoors|smiling|playing|reading|running|floating|resting|relaxing|[a-z]+ing)\b.*)?$')
_PLURAL_SUBJ = re.compile(r'\b(two|three|four|five|several|multiple|many|group of|pair of|couple|duo|trio|'
                          r'twins|girls|boys|women|men|characters|people)\b')
_PURE_GENERIC = re.compile(r'^anime illustration of (a|an) ([a-z\'-]+ ){0,3}([a-z-]+)$')
_GENERIC_BAD = re.compile(r'\b(with|in|wearing|sitting|standing|holding|lying|looking|on|at|from|and|posing|'
                          r'kneeling|leaning|walking|against|under|by|near|outdoors|indoors|smiling|two|three|'
                          r'of|the|shot|portrait|body|full|close-up|style|playing|reading|running|floating)\b')
GENERIC_NOUNS = {'girl', 'woman', 'lady', 'maiden', 'catgirl', 'foxgirl', 'schoolgirl', 'heroine', 'princess',
                 'witch', 'knight', 'warrior', 'nun', 'idol', 'student', 'vampire', 'maid', 'character', 'elf',
                 'succubus', 'mermaid', 'kunoichi', 'ninja', 'priestess', 'miko', 'android', 'angel', 'demon',
                 'gal', 'beauty', 'teenager', 'kemonomimi', 'catgirl-maid', 'bride', 'nurse', 'idol-girl',
                 'shrine-maiden', 'swordswoman', 'adventurer', 'mage', 'sorceress', 'cheerleader', 'gyaru',
                 'bunny-girl', 'dragon-girl', 'cat-girl', 'fox-girl', 'wolf-girl', 'oni', 'kitsune', 'robot',
                 'cyborg', 'pirate', 'soldier', 'officer', 'waitress', 'shopkeeper', 'gamer', 'artist'}


_POSITIONAL = re.compile(r'\b(top|bottom|upper|lower)[- ](left|right|corner|center|centre|edge|half|side)\b|'
                         r'\bon top of\b|\bat the top\b|\bthe top\b')


def _prep_for_facets(p):
    return re.sub(r'(\w)-style\b', r'\1', _POSITIONAL.sub(' ', _SHADE_BEFORE_COLOR.sub('', p)))


def facets_of(p):
    """Return the set of facet names matched by phrase p (lowercased, stripped)."""
    q = _prep_for_facets(p)
    out = {f for f, rx in FACET_RX.items() if rx.search(q)}
    if is_subject_like(p):
        out.add('subject')
    elif 'subject' in out and 'illustration' not in p:
        out.discard('subject')
    if 'eyes' in out:
        appear = bool(_EYE_APPEAR.search(p))
        state = bool(_EYE_STATE.search(p))
        if state and not (appear and re.search(r'\b(' + COLOR_WORDS + r')\b', p)):
            out.discard('eyes'); out.add('eyestate')
        elif not appear:
            out.discard('eyes'); out.add('eyestate')
    if 'hair' in out or 'hairacc' in out:
        rest = _HAIR_WORD_ONLY_ACC.sub(' ', p)
        rest = re.sub(r'\b(hairclips?|hairpins?|hairbands?|scrunchies?|barrettes?|kanzashi)\b', ' ', rest)
        if not FACET_RX['hair'].search(rest):
            out.discard('hair'); out.add('hairacc')
    if 'style' in out and _STRONG_STYLE.search(p):
        out.add('strongstyle')
    if _LIMB_LEAD.match(p):
        out.add('pose')
    if 'hair' in out and out & {'outfit', 'hairacc'}:
        rest = _HAIR_LOC.sub(' ', p)
        if not FACET_RX['hair'].search(rest):
            out.discard('hair'); out.add('hairacc')
    if 'hair' not in out and _HAIR2.search(p) and not (out & {'light', 'setting', 'outfit', 'style'}):
        out.add('hair')
    if _LIMBS.search(p) and not (out & {'outfit', 'hairacc', 'exposure', 'hair', 'eyes', 'pose'}):
        out.add('pose')
    if out & {'outfit', 'hairacc'}:
        q2 = _DECOR.sub(' ', _OUTFIT_COMPOUND.sub(' ', q))
        if 'setting' in out and not FACET_RX['setting'].search(q2):
            out.discard('setting')
        if 'light' in out and not FACET_RX['light'].search(_SHINY.sub(' ', q)):
            out.discard('light')
    if _PLAIN_ANIMAL_EARS.search(p):
        out.add('body')
        # "cat ears" / "fox tail" are body, not setting animals
        rest = _PLAIN_ANIMAL_EARS.sub(' ', p)
        if not FACET_RX['setting'].search(_prep_for_facets(rest)):
            out.discard('setting')
    return out


_SUBJ_SPLIT = re.compile(
    r'^((a|an|one|the)\s+)?([\w\'-]+\s+){0,3}?(girl|boy|woman|man|lady|maiden|catgirl|foxgirl|schoolgirl|heroine|'
    r'princess|witch|knight|warrior|nun|idol|student|vampire|female|figure|person|character|gal|beauty|maid|elf|'
    r'android|succubus|mermaid|kunoichi|ninja|priestess|miko)s?\s+(?:(?:who is|who|is|with|having)\s+)?'
    r'((with|in|wearing|sitting|standing|holding|lying|looking|on|at|posing|kneeling|leaning|walking|dressed|'
    r'clad|seen|shown|depicted|against|under|by|near|inside|outdoors|indoors|smiling|playing|reading|running|'
    r'floating|resting|relaxing|[a-z]+ing)\b.*)$')


def strip_subject_np(p):
    """'a girl with long silver hair' -> 'long silver hair'; 'girl sitting on a bench' -> 'sitting on a bench';
    a bare subject noun phrase ('a cat girl') -> ''. Phrases that are not subject-like are returned unchanged."""
    if not is_subject_like(p):
        return p
    m = _SUBJ_SPLIT.match(p)
    if not m or not m.group(5):
        return ''
    rest = re.sub(r"^(with|having)\s+", "", m.group(5)).strip()
    return rest


def is_subject_like(p):
    return bool(_SUBJ_NP.match(p)) or bool(re.match(r'^(anime |chibi )?(girl|woman)$', p))


def group_of(fs):
    """Group for a phrase with facet set fs. Hair / eye-colour phrases always go to APPEARANCE so that one donor
    owns the hair and eye description; otherwise SCENE (ACTION > PLACE) > OUTFIT > APPEARANCE. A phrase
    spanning action and place ('sitting on a sandy beach') is ACTION."""
    if 'subject' in fs:
        return SUBJ
    if 'hair' in fs or 'eyes' in fs:
        return APPEAR
    if 'style' in fs and not (fs & {'pose', 'outfit', 'exposure', 'setting'}):
        # 'high-contrast sunlight with heavy shadows' is lighting, 'soft lighting and cel shading' is style
        return PLACE if ('light' in fs and 'strongstyle' not in fs) else TAIL
    if fs & {'shot', 'pose', 'expr', 'eyestate'}:
        return ACTION
    if fs & {'light', 'setting'}:
        return PLACE
    if fs & {'outfit', 'exposure', 'hairacc'}:
        return OUTFIT
    if 'body' in fs:
        return APPEAR
    if fs & {'style', 'weakstyle'}:
        return TAIL
    return ACTION  # misc / unclassified


GROUP_FACETS = {APPEAR: ('hair', 'eyes', 'body'), OUTFIT: ('outfit', 'exposure', 'hairacc'),
                ACTION: ('shot', 'eyestate', 'expr', 'pose', 'light', 'setting'), PLACE: ('light', 'setting'),
                TAIL: ('style', 'weakstyle'),
                SUBJ: ('subject',)}


def primary_facet(fs, group=None):
    """Ordering / trimming facet of a phrase: the first facet (PRIMARY_ORDER) that belongs to its group."""
    if not fs:
        return 'misc'
    if group is None:
        group = group_of(fs)
    allowed = GROUP_FACETS[group]
    for f in PRIMARY_ORDER:
        if f in fs and f in allowed:
            return f
    return 'misc'


# ---------------------------------------------------------------- segmentation
_QUOTE = re.compile(r"(?<![a-z0-9])'[^']{0,200}?'(?![a-z0-9])|\"[^\"]{0,200}?\"")
_LEAD = re.compile(r'^(and|with|plus)\s+')
_COLOR_ONLY = re.compile(r'^(and\s+|or\s+)?((' + COLOR_WORDS + r'|lime|hot|neon|baby|sky|dusty|muted|electric|'
                         r'royal|forest|navy|mint|lemon|cherry|blood|wine|sea)(\s+|-)?){1,3}$')
_ENDS_COLOR = re.compile(r'\b(' + COLOR_WORDS + r')$')
_STARTS_COLOR = re.compile(r'^(and\s+|or\s+)?(' + COLOR_WORDS + r')\b')


def split_caption(text):
    """Quote-aware comma split; returns raw stripped fragments (lowercase)."""
    t = text.strip().lower()
    spans = []
    for m in _QUOTE.finditer(t):
        if ',' in m.group(0):
            spans.append((m.start(), m.end()))
    out, cur, i, si = [], [], 0, 0
    while i < len(t):
        if si < len(spans) and i == spans[si][0]:
            cur.append(t[spans[si][0]:spans[si][1]]); i = spans[si][1]; si += 1
            continue
        c = t[i]
        if c == ',':
            out.append(''.join(cur)); cur = []
        else:
            cur.append(c)
        i += 1
    out.append(''.join(cur))
    res = []
    for x in out:
        # sentence breaks inside a comma phrase ("... swing. she has long pink hair") become phrase breaks
        # (periods inside quotes are masked first)
        qs = []
        xm = _QUOTE.sub(lambda m: qs.append(m.group(0)) or f'\x00{len(qs) - 1}\x00', x)
        parts = [re.sub('\x00(\\d+)\x00', lambda m: qs[int(m.group(1))], y)
                 for y in re.split(r'\.\s+(?=[a-z\x00])|;\s+', xm)]
        for y in parts:
            y = re.sub(r'\s+', ' ', y).strip(' .;')
            if y:
                res.append(_pronoun_lead(y))
    return [y for y in res if y]


_PRON_LEAD = [(re.compile(r'^(she|the girl|the character) (has|have|had|features|shows)\s+'), ''),
              (re.compile(r'^(she|the girl|the character) (wears|is wearing|wore)\s+'), 'wearing '),
              (re.compile(r'^(she|the girl|the character) (holds|is holding)\s+'), 'holding '),
              (re.compile(r"^(she|the girl|the character) (is|was|appears|seems|'s)\s+"), ''),
              (re.compile(r'^(she|the girl|the character) (\w+s)\s+'), lambda m: m.group(2) + ' '),
              (re.compile(r'^the (background|scene|setting|backdrop) (features|shows|has|is|includes)\s+'), ''),
              (re.compile(r'^(there is|there are)\s+'), '')]


def _pronoun_lead(p):
    for rx, rep in _PRON_LEAD:
        q = rx.sub(rep, p, count=1)
        if q != p:
            return q.strip()
    return p


def segment(text, stats=None):
    """Split a caption into normalized phrases. Colour-list fragments ("a red, white, and blue dress") are merged
    back into one phrase; leading and/with/plus are stripped. stats (dict) collects merge/drop counts."""
    frags = split_caption(text)
    merged = []
    for f in frags:
        if merged and (_COLOR_ONLY.match(f) or (_ENDS_COLOR.search(merged[-1]) and _STARTS_COLOR.match(f)
                                                and len(merged[-1].split()) <= 4 and not
                                                FACET_RX['hair'].search(merged[-1]) and not
                                                FACET_RX['eyes'].search(merged[-1]))):
            merged[-1] = merged[-1] + ', ' + f
            if stats is not None:
                stats['merged_color'] = stats.get('merged_color', 0) + 1
            continue
        merged.append(f)
    # a merged phrase that still ends in a bare colour list ("wearing a red, white") is joined to the next one
    out = []
    i = 0
    while i < len(merged):
        p = merged[i]
        while (i + 1 < len(merged) and ', ' in p and _ENDS_COLOR.search(p.split(', ')[0]) and
               _COLOR_ONLY.match(p.split(', ')[-1])):
            p = p + ', ' + merged[i + 1]; i += 1
            if stats is not None:
                stats['merged_color'] = stats.get('merged_color', 0) + 1
        out.append(p); i += 1
    res = []
    pending = ''
    for p in out:
        q = _LEAD.sub('', p).strip()
        if not q:
            continue
        if pending:
            q = pending + ', ' + q
            pending = ''
        # a short fragment without any facet ('a bright') is the start of an adjective list: join the next one
        if len(q.split()) <= 3 and ', ' not in q and _DANGLING_ADJ.search(q) and not facets_of(q) \
                and p is not out[-1]:
            pending = q
            if stats is not None:
                stats['merged_dangling'] = stats.get('merged_dangling', 0) + 1
            continue
        # items of an enumeration ("stuffed animals including pandas, cats, birds") stay with their head phrase
        if res and len(q.split()) <= 2 and (_ENUM_HEAD.search(res[-1]) or res[-1] in _enum_open) \
                and not FACET_RX['hair'].search(q):
            res[-1] = res[-1] + ', ' + q
            _enum_open.add(res[-1])
            if stats is not None:
                stats['merged_enum'] = stats.get('merged_enum', 0) + 1
            continue
        if len(q.split()) == 1 and (not facets_of(q) or _LIMBS.fullmatch(q)):
            if stats is not None:
                stats['dropped_1word'] = stats.get('dropped_1word', 0) + 1
            continue
        res.append(q)
    _enum_open.clear()
    return res


_DANGLING_ADJ = re.compile(r'\b(bright|soft|warm|cool|smooth|clean|dark|light|pale|deep|vivid|vibrant|delicate|'
                           r'subtle|gentle|rich|muted|glossy|crisp|sharp|thin|bold|flat|detailed|high|low|very|'
                           r'slightly|highly|' + COLOR_WORDS + r')$')
_ENUM_HEAD = re.compile(r'\b(including|such as|like|featuring|consisting of|made of|mix of|variety of|such)\s+'
                        r'[^,]{1,30}$')
_enum_open = set()


def find_subject(phrases):
    """Return (subject_text, index_set_consumed). Subject = first phrase (among the first 3) containing
    'illustration'; a bare 'anime illustration' followed by a subject noun phrase ('a girl ...') is merged."""
    for i, p in enumerate(phrases[:3]):
        if 'illustration' in p:
            used = {i}
            subj = p
            if re.fullmatch(r'(anime |an anime |full[- ]body anime |full body anime )?illustration( style)?', p) \
                    and i + 1 < len(phrases) and is_subject_like(phrases[i + 1]):
                subj = 'anime illustration of ' + phrases[i + 1]
                used.add(i + 1)
            return subj, used
    return '', set()


def is_plural_subject(subj):
    s = re.sub(r'\bfrom\b.*$', '', subj)
    return bool(_PLURAL_SUBJ.search(s))


def is_pure_generic(subj):
    m = _PURE_GENERIC.match(subj)
    if not m:
        return False
    rest = subj[len('anime illustration of '):]
    if _GENERIC_BAD.search(rest.split(' ', 1)[1] if ' ' in rest else ''):
        return False
    noun = m.group(3)
    return noun in GENERIC_NOUNS or noun.endswith('girl')


# ---------------------------------------------------------------- tokenization for realization
STOP = {'a', 'an', 'the', 'of', 'on', 'in', 'at', 'to', 'for', 'from', 'by', 'with', 'her', 'his', 'its',
        'their', 'own', 'is', 'are', 'and', 'or', 'as', 'into', 'onto'}
# words dropped from a TAG's token pattern (prepositions stay: 'from above', 'on back', 'hand on hip')
TAG_DROP = {'a', 'an', 'the', 'her', 'his', 'its', 'their', 'own', 'is', 'are'}
BLOCKERS = {'with', 'and', 'or', 'but', 'while', 'featuring', 'plus', 'wearing', 'holding'}
SYN = {
    'gray': 'grey', 'silver': 'grey', 'silvery': 'grey', 'greyish': 'grey',  # not 'ash' ('ash-brown')
    'grayish': 'grey',
    'blond': 'yellow', 'blonde': 'yellow', 'golden': 'yellow', 'gold': 'yellow', 'platinum': 'yellow',
    'violet': 'purple', 'lavender': 'purple', 'lilac': 'purple', 'mauve': 'purple', 'purplish': 'purple',
    'crimson': 'red', 'scarlet': 'red', 'reddish': 'red', 'maroon': 'red', 'burgundy': 'red',
    'azure': 'blue', 'navy': 'blue', 'cobalt': 'blue', 'sapphire': 'blue', 'bluish': 'blue', 'cerulean': 'blue',
    'teal': 'aqua', 'turquoise': 'aqua', 'cyan': 'aqua',
    'emerald': 'green', 'jade': 'green', 'greenish': 'green',
    'amber': 'orange',
    'chestnut': 'brown', 'auburn': 'brown',
    'magenta': 'pink', 'fuchsia': 'pink', 'pinkish': 'pink',
    'smiling': 'smile', 'smiles': 'smile', 'smiled': 'smile',
    'blushing': 'blush', 'blushes': 'blush', 'blushed': 'blush',
    'sitting': 'sit', 'seated': 'sit', 'sits': 'sit',
    'standing': 'stand', 'stands': 'stand',
    'holding': 'hold', 'holds': 'hold', 'held': 'hold',
    'lying': 'lie', 'lies': 'lie', 'laying': 'lie',
    'kneeling': 'kneel', 'kneels': 'kneel',
    'looking': 'look', 'looks': 'look',
    'crying': 'cry', 'cries': 'cry',
    'squatting': 'squat', 'squats': 'squat',
    'leaning': 'lean', 'leans': 'lean',
    'crouching': 'crouch', 'crouched': 'crouch',
    'eared': 'ear', 'horned': 'horn', 'winged': 'wing',
    'pigtails': 'twintails', 'pigtail': 'twintails',
    'braided': 'braid', 'frilly': 'frilled', 'frills': 'frilled',
    'thighhighs': 'thighhigh', 'thighhigh': 'thighhigh',
    'catgirl': 'cat girl', 'foxgirl': 'fox girl', 'eyeglasses': 'glasses', 'spectacles': 'glasses',
}
NOSTEM = {'shorts', 'glasses', 'sunglasses', 'tights', 'pants', 'this', 'boots'}
_MULTI = [(re.compile(r'\btwin tails?\b'), 'twintails'), (re.compile(r'\bpony tails?\b'), 'ponytail'),
          (re.compile(r'\bthigh highs?\b'), 'thighhighs'), (re.compile(r'\bknee highs?\b'), 'kneehighs'),
          (re.compile(r'\bhair (clip|pin|band)s?\b'), r'hair\1'), (re.compile(r'\beye lashes\b'), 'eyelashes'),
          (re.compile(r'\bover the knee\b'), 'overknee'),
          (re.compile(r'\bhair (ornament|clip|pin|band|tie|ribbon|bow|flower|accessory|accessories|stick|tube|'
                      r'ring|bead|beads|scrunchie|bobble|cuff|ornaments|clips|pins|bands|ties|ribbons|bows|'
                      r'flowers|sticks|tubes|rings|scrunchies|bobbles|cuffs)\b'),
           lambda m: 'hair' + m.group(1).rstrip('s').replace('accessorie', 'accessory'))]


def raw_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace('-', ' ').replace("'s ", ' '))


def norm_token(w):
    w = SYN.get(w, w)
    if ' ' in w:
        return w
    if w in NOSTEM:
        return w
    if len(w) > 4 and w.endswith('ies'):
        w = w[:-3] + 'y'
    elif len(w) > 3 and w.endswith('s') and not w.endswith('ss') and not w.endswith('us'):
        w = w[:-1]
    return SYN.get(w, w)


def tokens(text):
    """Canonical token list for realization matching."""
    t = text.lower().replace('-', ' ')
    t = re.sub(r"'s\b", '', t)
    for rx, rep in _MULTI:
        t = rx.sub(rep, t)
    out = []
    for w in re.findall(r'[a-z0-9]+', t):
        n = norm_token(w)
        if ' ' in n:
            out.extend(n.split())
        else:
            out.append(n)
    return out


def tag_text(tag):
    """Tag string -> plain text (strip a parenthetical qualifier, underscores)."""
    t = re.sub(r'\s*\([^)]*\)\s*$', '', tag.replace('_', ' ')).strip()
    return t


def content_tokens(tag):
    return tuple(w for w in tokens(tag_text(tag)) if w not in TAG_DROP)


# ---------------------------------------------------------------- head-noun rule
# A noun tag written as the MODIFIER of a following noun is not realized: 'beach ball' is not a beach, 'shrine
# maiden' not a shrine, 'cat ears' not a cat, 'white shirt dress' not a white shirt. A following word that keeps the
# meaning (place / part nouns: 'beach background', 'library bookshelves', 'jacket pocket') does not block.
BASE_NOUNS = set("""lining crest logo emblem graphic gown toy keychain supply crossing bus zone ball towel umbrella parasol chair mat bag hat cap maiden ear tail horn wing uniform coat jacket
shirt dress skirt sweater vest boot shoe sock glove headband ribbon bow ornament clip pin print pattern motif
plush doll toy figure costume outfit girl boy woman man mask necklace earring bracelet ring choker belt
badge sticker sign poster book cup mug glass bottle can box case cover cushion pillow blanket rug curtain lamp
lantern light fan float ring tube suit gown apron sandal slipper heel strap kimono robe cape cloak scarf tie
sleeve collar flower petal leaf tree rose sword gun staff wand knife phone camera guitar bear cat dog bird fish
rabbit bunny fox wolf dragon horse frog duck chick sheep cow mouse squirrel deer penguin shark whale butterfly
star heart moon sun cloud ice cream cake candy cookie pie tea coffee juice milk water sea ocean wave sand shell
pin pendant charm tattoo""".split())
HEAD_OK = set("""background backdrop setting scene scenery view landscape environment interior exterior sky skyline
side area shore water sand floor ground wall window light lighting time atmosphere theme themed style aesthetic
collar hem pocket brim sleeve strap trim cuff button lining edge handle tip part table counter entrance door
doorway storefront terrace step stair path gate porch hallway corridor shelf bookshelf bookcase street road
building house room garden park field meadow grass tree branch leaf petal blossom flower outfit costume attire
clothes clothing uniform inspired""".split())
# time-of-day modifiers keep their meaning ('night city street' is at night)
TIME_MODS = {'night', 'nighttime', 'day', 'daytime', 'evening', 'morning', 'sunset', 'sunrise', 'dusk', 'dawn',
             'twilight', 'midnight', 'afternoon'}
VERBISH = set("""sit stand hold lie kneel look smile blush cry squat lean crouch run walk jump wink frown pout
open closed up down out off on over spread crossed raised wet messy shiny dark bright wavy curly straight sweat
sweaty sleepy smug happy sad angry tall small large huge long short medium very fluffy plump nude naked barefoot
topless see through solo multiple simple plain white black grey yellow red pink purple blue aqua green brown
orange blonde""".split())
NOUN_HEADS = set(BASE_NOUNS)


def set_noun_heads(extra):
    """Add data-derived noun heads (last token of frequent tags) to the head-noun rule."""
    NOUN_HEADS.clear()
    NOUN_HEADS.update(BASE_NOUNS)
    NOUN_HEADS.update(w for w in extra if w not in VERBISH and not w.endswith(('ing', 'ed', 'less', 'ful'))
                      and w not in STOP and w not in _PURE_COLOR_CANON and len(w) > 2 and not w.isdigit())


_PURE_COLOR_CANON = {norm_token(c) for c in _PURE_COLORS}


# the tag noun is part of a compound governed by an object: 'cat-shaped bag', 'cat print', 'school crest graphics',
# 'white cat stuffed toy', 'cat emblem'
DESCRIPTORS = set("""shaped print printed patterned pattern themed theme style styled motif design designed decal logo
emblem graphic crest insignia embroidered embroidery stuffed plush keychain charm sticker applique silhouette
mascot icon symbol""".split())
# 'maid outfit' is a maid, 'school uniform' is not a school
COSTUME_OK = {'outfit', 'costume', 'uniform', 'attire', 'clothes', 'clothing', 'inspired'}
COSTUME_HEADS = set("""maid nurse witch police idol cheerleader bunny sailor miko nun priestess ninja kunoichi pirate
knight samurai military waitress butler chef gothic lolita magical racing gym swim cowboy vampire angel demon
devil santa china""".split())
# time-of-day words count only when they qualify the scene ('night sky', 'night city street'), not an object
# ('evening gown', 'starry night-sky lining')
TIME_OK = set("""sky city cityscape street background backdrop scene scenery view time light lighting setting
landscape skyline market festival sea ocean beach forest garden park town road alley atmosphere cloud star moon
air breeze rain snow sun""".split())
COLLECTIVE_SENSE = {'school'}       # 'school of fish / dolphins'
WEATHER_OK = {'streak', 'drop', 'droplet', 'raindrop', 'puddle', 'flake', 'effect', 'particle', 'splash', 'ripple',
              'shower', 'storm', 'fall', 'cloud'}
HEAD_OK.update(WEATHER_OK)


def modifier_blocked(ct, ptoks, end):
    """True when the tag's head (ct[-1], at position end) is not the head of its noun group: it modifies a
    following noun ('beach ball'), is part of an object compound ('cat-shaped bag', 'cat stuffed toy'), or is a
    time word inside an object description ('night-sky lining')."""
    head = ct[-1]
    if head in COLLECTIVE_SENSE and end + 1 < len(ptoks) and ptoks[end + 1] == 'of':
        return True                  # 'schools of fish' is not a school
    is_time = head in TIME_MODS
    if not is_time and (head in VERBISH or head.endswith('ing')):
        return False
    if not is_time and head not in NOUN_HEADS:
        return False
    chain = []                       # the nouns that follow in the same noun group
    j = end + 1
    while j < len(ptoks) and j <= end + 4:
        w = ptoks[j]
        if w in DESCRIPTORS:
            return True
        if w in _PURE_COLOR_CANON:          # 'sky blue dress', 'night-sky blue cuffs'
            j += 1
            continue
        if w in NOUN_HEADS or w in HEAD_OK or w in TIME_OK:
            chain.append(w)
            j += 1
            continue
        break
    if not chain:
        return False
    if is_time:                      # 'night sky' yes, 'night-sky lining' / 'evening gown' no
        return any(w not in TIME_OK for w in chain)
    for w in chain:                  # any object noun in the chain governs: 'shrine maiden outfit', 'cat ears headband'
        if w in COSTUME_OK:
            if head not in COSTUME_HEADS:
                return True
        elif w not in HEAD_OK:
            return True
    return False


def match_tokens(ct, ptoks, pos=None, head_rule=True):
    """Ordered-window lexical match: content tokens ct appear in order in ptoks, consecutive gap <= 3 tokens,
    no blocker word (with/and/...) in between, and (head_rule) the tag's head noun is not the modifier of a
    following noun."""
    if not ct:
        return False
    if pos is None:
        pos = {}
        for i, w in enumerate(ptoks):
            pos.setdefault(w, []).append(i)
    for w in ct:
        if w not in pos:
            return False
    for start in pos[ct[0]]:
        cur = start
        ok = True
        for w in ct[1:]:
            nxt = None
            for j in pos[w]:
                if cur < j <= cur + 4:
                    if any(ptoks[k] in BLOCKERS for k in range(cur + 1, j)):
                        continue
                    nxt = j
                    break
            if nxt is None:
                ok = False
                break
            cur = nxt
        if ok and not (head_rule and modifier_blocked(ct, ptoks, cur)):
            return True
    return False


# ---------------------------------------------------------------- backdrop (single location)
LOCATION_WORDS = set("""background backdrop scenery setting landscape surroundings environment room bedroom
classroom library cafe restaurant kitchen bathroom office street alley alleyway city cityscape town village forest
wood jungle beach shore seaside ocean sea park garden field meadow shrine temple church cathedral castle palace
station train pool poolside rooftop balcony hallway corridor stage desert mountain lake river interior ruin
battlefield arena stadium gym school hospital bar shop store market attic basement cave island underwater
library bookstore greenhouse conservatory onsen bath bathhouse courtyard plaza harbor port pier dock farm
countryside valley cliff waterfall swamp lab laboratory throne spaceship cockpit dungeon tavern inn outdoors
outdoor outside indoors indoor building skyline skyscraper rooftops bedroom pavilion road sidewalk crosswalk
highway bridge pier boardwalk subway platform""".split())


def is_backdrop(text):
    """A phrase that names the overall background / location ('tropical beach background', 'inside a dark
    gothic library', 'plain white background'), not one using a place word as a modifier ('beach ball')."""
    toks = tokens(text)
    for i, w in enumerate(toks):
        if w in LOCATION_WORDS and not modifier_blocked((w,), toks, i) and \
                not (i + 1 < len(toks) and toks[i + 1] in _GARMENT_NEXT):        # 'school uniform', 'gym shorts'
            return True
    return False


_GARMENT_NEXT = set('uniform outfit costume attire clothes clothing shorts swimsuit shirt skirt dress jacket shoe '
                    'sneaker sock bag hat cap towel'.split())


def gram_positions(toks, nmax=3):
    """(gram, end position) for all 1..nmax grams not starting / ending in a stop word."""
    out = []
    L = len(toks)
    for n in range(1, nmax + 1):
        for i in range(L - n + 1):
            g = toks[i:i + n]
            if g[0] in STOP or g[-1] in STOP:
                continue
            out.append((' '.join(g), i + n - 1))
    return out


def ngrams(toks, nmax=3):
    out = set()
    L = len(toks)
    for n in range(1, nmax + 1):
        for i in range(L - n + 1):
            g = toks[i:i + n]
            if g[0] in STOP or g[-1] in STOP:
                continue
            out.add(' '.join(g))
    return out


def has_gram(ptoks_joined, gram):
    return (' ' + gram + ' ') in (' ' + ptoks_joined + ' ')


# ---------------------------------------------------------------- text / marks: overlay vs scene text
# OVERLAY = marks on the picture itself (watermark, artist signature, logo / text in a corner, handles, copyright);
# SCENE = text on an object inside the scene (shirt print, sign, book cover, tattoo, speech bubble). Rules chosen on
# the donor phrases and checked on hand-labelled phrases.
OVERLAY, SCENE_TEXT = 'overlay', 'scene'
_MARK_PRE = re.compile(r"watermark|signature|signed|artist|credit|logo|text|letter|writ|typograph|calligraph|font|"
                       r"kanji|katakana|hiragana|hangul|cyrillic|characters|caption|title|label|read|says|saying|"
                       r"inscri|copyright|©|rights reserved|username|handle|@|url|website|\.com|patreon|pixiv|twitter|"
                       r"weibo|fanbox|skeb|instagram|tumblr|deviantart|bilibili|lofter|mihuashi|youtube|bubble|balloon|"
                       r"dialogue|sound effect|onomatopoeia|sfx|qr code|barcode|dated|number|numeral|digit|slogan|"
                       r"initials|monogram|autograph|printed|embroidered|stitched|engraved")
_QUOTED = re.compile(r"(?<![a-z0-9])'[^']{0,200}?'(?![a-z0-9])|\"[^\"]{0,200}?\"|(?<![a-z0-9])'[^']{0,200}$")
_SIG_NOUN_NEXT = (r'(?=\s*$|\s*,|\s+(in|on|at|across|along|near|to|below|above|beside|next|under|over|and|reading|text|'
                  r'mark|marks|scribble|element|elements|visible|written|placed|located|stamp|logo|watermark|of the|'
                  r'faintly|partially|hidden|drawn|signed)\b)')
_INHERENT = re.compile(
    r"\bwatermarks?\b|\bsigned\b|\b(artist'?s?|character|copyright|notice) signatures?\b|"
    r"\bsignatures?" + _SIG_NOUN_NEXT + r"|\bartists?'?s? (names?|marks?|logos?|handles?|credits?|tags?|initials|"
    r"monograms?|stamps?|seals?|ids?|usernames?|links?|info)\b|\bcopyright(ed)?\b|©|"
    r"\ball rights? reserved\b|\busernames?\b|"
    r"\b(twitter|x|social media|instagram|pixiv|weibo|bilibili|patreon|youtube|tiktok)\s+(handles?|ids?|usernames?|"
    r"accounts?|names?|links?|logos?|icons?)\b|(?<![\w.])@[a-z0-9_]|\burls?\b|\bwebsites?\b|\bweb ?address(es)?\b|"
    r"\.com\b|\b(patreon|pixiv|fanbox|skeb|weibo|tumblr|deviantart|bilibili|lofter|mihuashi|miyoushe|twitter)\b")
_BUBBLE = re.compile(r'\b(speech|thought|dialogue|dialog|word|chat) (bubbles?|balloons?|box(es)?)\b|\bdialogue\b|'
                     r'\bspeech balloons?\b|\bbubbles? of (text|dialogue|speech)\b')
# picture positions, also compound and coordinated ones: 'the mid-left', 'the center-left', 'the center and bottom',
# 'the top and bottom edges', 'at top' (no article only before a position word that is not followed by 'of')
_POS_WORD = (r'(top|bottom|upper|lower|left|right|center|centre|middle|mid|corner|corners|edge|edges|side|sides|'
             r'border|borders|margin|margins|half|area|portion)')
_POS_SEQ = _POS_WORD + r'((-| )' + _POS_WORD + r')*'
_POS_COORD = _POS_SEQ + r'(,? (and|or|&) (the )?' + _POS_SEQ + r')*'
_POS_PREP = r'\b(in|at|on|along|across|near|to|toward|towards|from|around)'
_POSITION = re.compile(_POS_PREP + r' the (' + _POS_COORD + r'|(background|foreground|image|picture|frame|scene|'
                       r'canvas|page|composition|illustration)((-| )(' + _POS_SEQ + r'|background))*)\b|' +
                       r'\bat (top|bottom)\b(?! of\b)|'
                       r'\bcorner\b|\boverlaid\b|\boverlay\b|\bsemi-transparent\b|\bfaint(ly)?\b')
_ACTIVITY = re.compile(r"(^|\b(is|while|and|as|sits|sitting|lies|lying|quietly|calmly|currently|busy|engrossed in)\s+)"
                       r"reading\b|\breading (a|an|the|her|his|their|its|some|from|aloud|quietly|together|manga|books?|"
                       r"magazines?|newspapers?|letters?|scrolls?|documents?|novels?|comics?|notes?|papers?|tablets?|"
                       r"glasses)\b|\b(used )?for writing\b|\bwriting (in|on) (a|an|her|his)\b|"
                       r"\bcalligraphy (brush|brushes|pens?|sets?)\b")
_TEXT_WORD = re.compile(r"\b(texts?|lettering|letterings|typography|typographic|calligraphy|calligraphic|fonts?|"
                        r"kanji|katakana|hiragana|hangul|cyrillic|(japanese|chinese|korean|english|latin|russian|"
                        r"arabic|thai) (characters|script|writing|letters|words)|logos?|captions?|title|titled|"
                        r"words?|wording|writing|written|inscriptions?|inscribed|numbers|"
                        r"number \d+|numerals?|digits|"
                        r"sound effects?|onomatopoeia|sfx|slogans?|initials|monograms?|autograph|letters?\s+Q|"
                        r"(?<!style )labels?|"
                        r"credits?|(character|series|game|artist|copyright|company|title) names?|"
                        r"(" + COLOR_WORDS + r"|faint|large|small|big|bold|neon|glowing|printed|stylized|pixel|"
                        r"pixelated|block|capital|cursive|handwritten|floating) letters)\b")
_TEXT_VERB = re.compile(r"\b(reading|reads|says|saying|spelling|spells|labeled|labelled|labels?|numbered|printed|"
                        r"embroidered|"
                        r"stitched|engraved|emblazoned|branded)\b")
_TEXT_VERB_BARE = re.compile(r'\b(reading|says|saying|spelling|labell?ed|numbered)\b')    # text even without a quote
_QR = re.compile(r'\b(qr codes?|barcodes?|dated|date stamps?)\b')
_CARRIER = re.compile(
    r"\b(\w*shirts?|tee|tees|(?<!the )(?<!and )tops?\b(?! *(left|right|center|centre|middle|corner|edge|half|and|"
    r"of|,|$))|\w*hoodies?|\w*sweaters?|sweatshirts?|jackets?|coats?|jerseys?|dress(es)?|skirts?|miniskirt|shorts|"
    r"pants|jeans|leggings|pantyhose|tights|socks?|stockings|thighhighs|shoes?|sneakers?|clocks?|watch(es)?|boots?|"
    r"soles?|caps?|hats?|beret|visor|helmet|headband|hairband|hair ?clips?|clips?|hair ?pins?|pins?|ribbons?|"
    r"bows?|ties?|necktie|collars?|chokers?|armbands?|wristbands?|belts?|straps?|waistband|sash|aprons?|bib|uniforms?|"
    r"swimsuits?|bikinis?|bras?|camisole|underwear|panties|leotard|bodysuit|sleeves?|patch(es)?|badges?|buttons?|"
    r"tags?|name ?tags?|lanyards?|bags?|backpacks?|pouch|cases?|masks?|gloves?|scarf|scarves|cape|cloak|vests?|"
    r"overalls|pockets?|hood|blazer|cardigan|pajamas|kimono|robe|tracksuit|windbreaker|parka|pendants?|necklaces?|"
    r"bracelets?|keychains?|charms?|pom ?poms?|outfit|costume|clothes|clothing|garments?|fabric|cloth|"
    r"signs?|signboards?|signposts?|plaques?|placards?|nameplates?|billboards?|posters?|banners?|flags?|\w*books?|"
    r"manga|magazines?|covers?|pages?|papers?|"
    r"(paper|sticky|post-it|memo|handwritten) notes?|cards?|envelopes?|tickets?|receipts?|maps?|scrolls?|menus?|"
    r"\w*boards?|slates?|screens?|monitors?|displays?|phones?|smartphones?|laptops?|tablets?|tv|television|"
    r"computers?|cups?|mugs?|cans?|bottles?|jars?|"
    r"(?<!text )boxe?s?|cartons?|packages?|packets?|wrappers?|stickers?|tape|pillows?|cushions?|blankets?|"
    r"(product|price|shipping|warning|nutrition|bottle|jar|can|luggage|name) labels?|"
    r"towels?|umbrellas?|parasols?|fans?|lanterns?|balloons?|kites?|cakes?|plates?|trays?|pianos?|guitars?|drums?|"
    r"drumsticks|microphones?|megaphones?|speakers?|walls?|tiles?|tiled|doors?|windows?|storefronts?|shops?|"
    r"stores?|stalls?|buildings?|rooftops?|towers?|bus|train|cars?|trucks?|vending machines?|machines?|gates?|fences?|"
    r"crates?|barrels?|chairs?|desks?|tables?|benches?|lockers?|picture frames?|photo frames?|polaroids?|photos?|"
    r"photographs?|tattoos?|chest|breasts?|arms?|forearms?|shoulders?|thighs?|legs?|cheeks?|forehead|hands?|palms?|"
    r"wrists?|neck|belly|stomach|skin|weapons?|swords?|guns?|rifles?|shields?|armou?r|blades?|staff|wand|dolls?|"
    r"plush\w*|toys?|figures?|mascot costume|bouquet|gift|letter(?!\s+Q)|postage stamps?|"
    r"bands?|thigh-highs?|knee-highs?)\b(?!-|\s+(style|styled|themed|inspired)\b)|"
    r"\b(printed|embroidered|stitched|engraved|emblazoned|branded|painted on|written on)\b")
# explicit evidence that text sits on the picture itself; it wins over carrier words ("magazine-style graphic design
# text overlays across the image", "printed text overlay texture", "smartphone camera interface overlay with text")
_OVER_EVIDENCE = re.compile(r"\b(texts?|typography|lettering|graphics?|logos?|interface|ui|card|captions?)\s+(\w+\s+)?"
                            r"overlays?\b|\boverlaid\b|\bacross the (whole |entire )?(image|picture|frame|canvas|"
                            r"composition|scene)\b|\bover the (whole |entire )?(image|picture)\b|\bgraphic design\b")
# 'on her / the X' names an object X, never a picture position ('on the mid-left', 'on the far right')
_ON_OBJECT = re.compile(r'\bon (her|his|the|a|an|its|their) (?!(' + _POS_WORD[1:-1] + r'|background|foreground|image|'
                        r'picture|frame|scene|canvas|page|composition|illustration|other|far)\b)[a-z]')


def text_mark(p):
    """'overlay' (a mark on the picture itself: watermark, artist signature, logo / text in a corner, handle,
    copyright), 'scene' (text on something inside the scene: shirt print, sign, book cover, tattoo, label, speech
    bubble) or None (no text: 'peace sign', 'reading a book', 'umbrella handle', 'her signature dress')."""
    if not _MARK_PRE.search(p):
        return None
    q = _QUOTED.sub(' Q ', p)
    if _INHERENT.search(q):
        return OVERLAY
    if _BUBBLE.search(q):
        return SCENE_TEXT
    q = _ACTIVITY.sub(' ', q)
    text = bool(_TEXT_WORD.search(q))
    verb = bool(_TEXT_VERB.search(q)) and ('Q' in q.split() or bool(_TEXT_VERB_BARE.search(q)))
    qr = bool(_QR.search(q))
    if (text or verb or qr) and _OVER_EVIDENCE.search(q):
        return OVERLAY              # explicit evidence wins over carrier words
    if qr:
        if _POSITION.search(q) and not _CARRIER.search(_POSITION.sub(' ', q)):
            return OVERLAY
        return SCENE_TEXT if _CARRIER.search(_POSITION.sub(' ', q)) else None
    if not text and not verb:
        return None
    q = _SIDE_OF_OBJECT.sub(r'\1 the ', q)      # 'on the right chest' is on the chest, not on the right of the image
    rest = _POSITION.sub(' ', q)
    on_image = bool(_ON_IMAGE.search(q))
    if _carried(rest, on_image) or _ON_OBJECT.search(rest) or (not on_image and _WITH_LABEL.search(rest)):
        return SCENE_TEXT
    # no carrier: text words are on the picture; 'X reading ...' too when placed on the picture ('date reading ...
    # on the left'), else X is an unlisted object ('billiard balls numbered 2')
    return OVERLAY if text or on_image else SCENE_TEXT


# a carrier is attached to the text: a preposition before it ('on her chest', 'across the front of the shirt'), a
# text verb / 'with' after it ('a shirt with a logo', 'sign reading ...'), or a text word next to it ('sticker text',
# 'logo patch'). Needed whenever the phrase also places the text on the picture ('... in the top left') and always for
# words that are often figurative ('metal band logo', 'a bow icon', 'a card graphic')
_ATTACH_BEFORE = re.compile(r'\b(on|onto|across|along|over|upon|under|inside|in|into|from|of)\s+((her|his|the|a|an|'
                            r'its|their|one|both|each|this)\s+)?([\w-]+\s+){0,3}$')
_ATTACH_AFTER = re.compile(r"\s+([\w-]+\s+)?(with|reading|reads|labell?ed|saying|says|featuring|bearing|"
                           r"displaying|showing|printed|embroidered|stitched|engraved|written|numbered|that reads)\b")
_TEXT_NEXT = re.compile(r'\s+(texts?|lettering|logos?|kanji|writing|written|numbers?|numbered|labels?|title|'
                        r'print|printed|graphic|slogans?|Q)\b')
# an object with labels ('pork knuckle with labels', 'bottles with handwritten labels'): labels need such an object
_WITH_LABEL = re.compile(r'\b(?!(texts?|lettering|writing|typography|graphics?|elements?|lines?|arrows?)\b)[a-z]+ '
                         r'with\s+([\w-]+\s+){0,3}labels?\b')
_TEXT_BEFORE = re.compile(r'\b(texts?|lettering|logos?|kanji|writing|lettered|numbered|title|caption|Q)\s+$')
_AMBIGUOUS = re.compile(r'(manga|bands?|bows?|cards?|fans?|staff|figures?|covers?|cases?|displays?|buttons?|seals?|'
                        r'hands?|arms?|legs?|neck|skin|chest)')
# on-image positions that require an attached carrier (depth words 'in the background', 'faint', 'on the side' do not)
_ON_IMAGE = re.compile(_POS_PREP + r' the (?!sides?\b)(' + _POS_COORD + r'|image|picture|frame|canvas|composition|'
                       r'illustration)\b|\bat (top|bottom)\b(?! of\b)|\bcorner\b')
_SIDE_OF_OBJECT = re.compile(r'\b(on|at|in|across|along) the (left|right|upper|lower|top|bottom|front|back)\s+'
                             r'(?=(chest|breast|shoulder|sleeve|arm|forearm|thigh|leg|cheek|hip|side of|collar|lapel|'
                             r'pocket|cuff|wrist|hand|ear|skirt|strap|shirt|jacket|hoodie|cap|hat)s?\b)')


def _carried(rest, on_image):
    """Does a carrier object hold the text (rest: the phrase without its on-image positions)?"""
    for m in _CARRIER.finditer(rest):
        before, after = rest[:m.start()], rest[m.end():]
        ambiguous = bool(_AMBIGUOUS.fullmatch(m.group(0)))
        if not on_image and not ambiguous:
            return True
        if _ATTACH_BEFORE.search(before) or _ATTACH_AFTER.match(after):
            return True
        # a text word next to it ('sticker text', 'logo patch'); not for figurative words ('band logo', 'bow icon')
        if not ambiguous and (_TEXT_NEXT.match(after) or _TEXT_BEFORE.search(before)):
            return True
    return False


# a constraint that ASKS for a mark (its realizing phrase is never filtered): Danbooru text / mark tags
MARK_TAG_RX = re.compile(r'\b(signature|signed|watermarks?|watermark grid|usernames?|logos?|text|copyright|dated|'
                         r'speech bubbles?|thought bubbles?|qr code|barcode|sound effects|onomatopoeia|pixiv id|'
                         r'web address|(artist|character|company|song|commissioner|group|circle|copyright) name|title|'
                         r'name tag|lettering|kanji|translated|commission)\b')


def is_mark_request(text):
    """Does a constraint text ask for a text mark (signature, watermark, english text, speech bubble, ...)?"""
    return bool(MARK_TAG_RX.search(text)) or text_mark(text) is not None


IMPLICIT_TAGS = {'1girl', 'solo'}
# a non-subject phrase that names the subject again ('full shot of a catgirl walking ...') is not reused by donors
SUBJ_MENTION = re.compile(r"(^|\b(a|an|the|of)\s+)([\w-]+\s+){0,2}?(girl|catgirl|foxgirl|woman|schoolgirl|boy|man|"
                          r"lady|cat girl|fox girl|wolf girl|dog girl|demon girl|bunny girl)s?\b(?!'s|\s+(ears?|"
                          r"hair|tail|headband|hat|ear))")
COUNT_TAGS = {'1girl', '1boy', '1other'}
PLURAL_COUNT_TAGS_RX = re.compile(r'^(\d+(girls|boys|others)|multiple (girls|boys|others)|6\+(girls|boys))$')
