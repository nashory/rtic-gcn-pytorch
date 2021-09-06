__VERBOSE__ = False


class SpellChecker:
    @classmethod
    def correct_token(cls, token):
        prev_token = token
        if token == "seethru":
            token = "see-through"
        elif token == "spagetti":
            token = "spaghetti"
        elif token == "colourfull":
            token = "colourful"
        elif token == "checkered":
            token = "checked"
        elif token == "colorful":
            token = "colourful"
        elif token == "multicolored":
            token = "multicoloured"
        elif token == "coloring":
            token = "colouring"
        elif token == "lighgter":
            token = "lighter"
        elif token == "ssleeves":
            token = "sleeves"
        elif token == "cirlces":
            token = "circles"
        elif token == "plumpkin":
            token = "pumpkin"
        elif token == "lettters":
            token = "letters"
        elif token == "charchol":
            token = "charcoal"
        elif token == "dwon":
            token = "down"
        elif token == "karger":
            token = "larger"
        elif token == "lighther":
            token = "lighter"
        elif token == "darler":
            token = "darker"
        elif token == "diffrent":
            token = "different"
        elif token == "sleevs":
            token = "sleeves"
        elif token == "cicular":
            token = "circular"
        elif token == "sligthtly":
            token = "slightly"
        elif token == "backgtound":
            token = "background"
        elif token == "shirter":
            token = "shorter"
        elif token == "sleees":
            token = "sleeves"
        elif token == "buttons0":
            token = "buttons"
        elif token == "qhite":
            token = "white"
        elif token == "logner":
            token = "longer"
        elif token == "wiith":
            token = "with"
        elif token == "sleve":
            token = "sleeve"
        elif token == "yaris":
            token = "paris"
        elif token == "stylis":
            token = "stylish"
        elif token == "daker":
            token = "darker"
        elif token == "idential":
            token = "identical"
        elif token == "colllar":
            token = "collar"
        elif token == "grapic":
            token = "graphic"
        elif token == "organge":
            token = "orange"
        elif token == "blac":
            token = "black"
        elif token == "diagnal":
            token = "diagonal"
        elif token == "smalled":
            token = "smaller"
        elif token == "innappropriate":
            token = "inappropriate"
        elif token == "sinlge":
            token = "single"
        elif token == "ehite":
            token = "white"
        elif token == "hoddie":
            token = "hoodie"
        elif token == "balck":
            token = "black"
        elif token == "colorufl":
            token = "colourful"
        elif token == "camoflage":
            token = "camouflage"
        elif token == "adarker":
            token = "darker"
        elif token == "sleeces":
            token = "sleeves"
        elif token == "checkered":
            token = "checked"
        elif token == "hawaiin":
            token = "hawaiian"
        elif token == "simplier":
            token = "simpler"
        elif token == "wpockets":
            token = "pockets"
        elif token == "simle":
            token = "simple"
        elif token == "pocke":
            token = "pocket"
        elif token == "greay":
            token = "grey"
        elif token == "ligher":
            token = "lighter"
        elif token == "shiney":
            token = "shiny"
        elif token == "mofe":
            token = "more"
        elif token == "coloful":
            token = "colourful"
        elif token == "grayer":
            token = "grey"
        elif token == "desgin":
            token = "design"
        elif token == "bllue":
            token = "blue"
        elif token == "blousy":
            token = "blouse"
        elif token == "necklin":
            token = "neckline"
        elif token == "waiste":
            token = "waist"
        elif token == "grpahic":
            token = "graphic"
        elif token == "shortr":
            token = "shorter"
        elif token == "writng":
            token = "writing"
        elif token == "seethru":
            token = "seethough"
        elif token == "transparentit":
            token = "transparent"
        elif token == "lwaisted":
            token = "waisted"
        elif token == "torquose":
            token = "turquoise"
        elif token == "blalck":
            token = "black"
        elif token == "sleevers":
            token = "sleeves"
        elif token == "vnecked":
            token = "v-neck"
        elif token == "fashioanble":
            token = "fashionable"
        elif token == "delihtful":
            token = "delightful"
        elif token == "shojrter":
            token = "shorter"
        elif token == "revealaing":
            token = "revealing"
        elif token == "ruffels":
            token = "ruffles"
        elif token == "rvealing":
            token = "revealing"
        elif token == "waisr":
            token = "waist"
        elif token == "reveailing":
            token = "revealing"
        elif token == "recealing":
            token = "revealing"
        elif token == "paterned":
            token = "patterned"
        elif token == "lighterer":
            token = "lighter"
        elif token == "spghetti":
            token = "spaghetti"
        elif token == "beltess":
            token = "beltless"
        elif token == "londer":
            token = "longer"
        elif token == "texured":
            token = "textured"
        elif token == "revaeling":
            token = "revealing"
        elif token == "pattterned":
            token = "patterned"
        elif token == "folral":
            token = "floral"
        elif token == "buttos":
            token = "buttons"
        elif token == "necline":
            token = "neckline"
        elif token == "longsleev":
            token = "longsleeve"
        elif token == "speggeti":
            token = "spaghetti"
        elif token == "darder":
            token = "darker"
        elif token == "ashirt":
            token = "shirt"
        elif token == "casuall":
            token = "casual"
        elif token == "shorer":
            token = "shorter"
        elif token == "torquiose":
            token = "turquoise"
        elif token == "sleevelss":
            token = "sleeveless"
        elif token == "pattrn":
            token = "pattern"
        elif token == "contrastic":
            token = "constrast"
        elif token == "dront":
            token = "front"
        elif token == "frell":
            token = "frill"
        elif token == "tshirts":
            token = "t-shirt"
        elif token == "graphi":
            token = "graphic"
        elif token == "shufflin":
            token = "shuffling"
        elif token == "cheched":
            token = "checked"
        elif token == "whitewith":
            token = "white"
        elif token == "textered":
            token = "textured"
        elif token == "grpahics":
            token = "graphics"
        elif token == "elss":
            token = "less"
        elif token == "horzontally":
            token = "horizontally"
        elif token == "haedphone":
            token = "headphone"
        elif token == "printim":
            token = "printing"
        elif token == "weaable":
            token = "wearable"
        elif token == "religoius":
            token = "religious"
        elif token == "graffic":
            token = "graphic"
        elif token == "millitary":
            token = "military"
        elif token == "diagnol":
            token = "diagonal"
        elif token == "smalelr":
            token = "smaller"
        elif token == "letterng":
            token = "lettering"
        elif token == "constrast":
            token = "contrast"
        elif token == "hoody":
            token = "hoodie"
        elif token == "similer":
            token = "similar"
        elif token == "kintted":
            token = "knitted"
        elif token == "leathre":
            token = "leather"
        elif token == "inmage":
            token = "image"
        elif token == "navey":
            token = "navy"
        elif token == "sophisticaed":
            token = "sophisticated"
        elif token == "ptint":
            token = "print"
        elif token == "differnt":
            token = "different"
        elif token == "soulders":
            token = "shoulder"
        elif token == "reviealing":
            token = "revealing"
        elif token == "yeellow":
            token = "yellow"
        elif token == "bicyle":
            token = "bicycle"
        elif token == "burgendy":
            token = "burgundy"
        elif token == "desighn":
            token = "design"
        elif token == "contrastiing":
            token = "contrast"
        elif token == "necj":
            token = "neck"
        elif token == "casaul":
            token = "casual"
        elif token == "wwhite":
            token = "white"
        elif token == "strpies":
            token = "stripe"
        elif token == "andlonger":
            token = "longer"
        elif token == "staps":
            token = "straps"
        elif token == "sleave":
            token = "sleeve"
        elif token == "orane":
            token = "orange"
        elif token == "colr":
            token = "color"
        elif token == "lenth":
            token = "length"
        elif token == "traingle":
            token = "triangle"
        elif token == "fleeves":
            token = "sleeve"
        elif token == "sleevd":
            token = "sleeve"
        elif token == "whinte":
            token = "white"
        elif token == "organe":
            token = "orange"
        elif token == "collers":
            token = "collars"
        elif token == "ablack":
            token = "black"
        elif token == "collored":
            token = "coloured"
        elif token == "yelloe":
            token = "yellow"
        elif token == "teeshirt":
            token = "t-shirt"
        elif token == "loger":
            token = "longer"
        elif token == "revaling":
            token = "revealing"
        elif token == "snall":
            token = "small"
        elif token == "stariht":
            token = "straight"
        elif token == "nlack":
            token = "black"
        elif token == "ligther":
            token = "lighter"
        elif token == "sleaves":
            token = "sleeve"
        elif token == "clored":
            token = "coloured"
        elif token == "sholder":
            token = "shoulder"
        elif token == "draker":
            token = "darker"
        elif token == "assymetrical":
            token = "assymetrical"
        elif token == "floorlenght":
            token = "floor-length"
        elif token == "twopiece":
            token = "two-piece"
        elif token == "longsleeve":
            token = "long-sleeved"
        elif token == "sholders":
            token = "shoulders"
        elif token == "lonf":
            token = "long"
        elif token == "highwaisted":
            token = "high-waisted"
        elif token == "shouldersand":
            token = "shoulders"
        elif token == "twotoned":
            token = "two-tone"
        elif token == "productut":
            token = "product"
        elif token == "puple":
            token = "purple"
        elif token == "sleevless":
            token = "sleeveless"
        elif token == "sleevleess":
            token = "sleeveless"
        elif token == "animallike":
            token = "animal-like"
        elif token == "patterened":
            token = "patterned"
        elif token == "reavealing":
            token = "revealing"
        elif token == "scouped":
            token = "scooped"
        elif token == "sheerer":
            token = "sheer"
        elif token == "briter":
            token = "brighter"
        elif token == "longsleeves":
            token = "long-sleeved"
        elif token == "longsleeved":
            token = "long-sleeved"
        elif token == "shorteer":
            token = "shorter"
        elif token == "seethrough":
            token = "see-through"
        elif token == "romperlike":
            token = "romperlike"
        elif token == "lenght":
            token = "length"
        elif token == "sleevees":
            token = "sleeves"
        elif token == "patterend":
            token = "patterned"
        elif token == "asymettrical":
            token = "asymmetrical"
        elif token == "colorblock":
            token = "colorblock"
        elif token == "haltered":
            token = "halterneck"
        elif token == "asymetrical":
            token = "asymmetrical"
        elif token == "cleveage":
            token = "cleavage"
        elif token == "plaidlike":
            token = "plaidlike"
        elif token == "feminsit":
            token = "feminist"
        elif token == "deisgn":
            token = "design"
        elif token == "thiner":
            token = "thinner"
        elif token == "femineme":
            token = "feminine"
        elif token == "coldshoulder":
            token = "coldshoulder"
        elif token == "sleveless":
            token = "sleeveless"
        elif token == "faminine":
            token = "feminine"
        elif token == "stras":
            token = "straps"
        elif token == "pinkyellow":
            token = "pinkyellow"
        elif token == "thqt":
            token = "that"
        elif token == "turqoise":
            token = "turquoise"
        elif token == "shortsleeve":
            token = "short-sleeved"
        elif token == "colorfull":
            token = "colourful"
        elif token == "kneww":
            token = "knew"
        elif token == "sweaterlike":
            token = "sweater"
        elif token == "onesleeved":
            token = "single-sleeve"
        elif token == "tangtop":
            token = "tanktop"
        elif token == "kneelength":
            token = "knee-length"
        elif token == "oart":
            token = "part"
        elif token == "neckine":
            token = "neckline"
        elif token == "kneee":
            token = "knees"
        elif token == "blacl":
            token = "black"
        elif token == "drss":
            token = "dress"
        elif token == "assymetric":
            token = "asymmetric"
        elif token == "prdouct":
            token = "product"
        elif token == "colur":
            token = "colour"
        elif token == "comapre":
            token = "compare"
        elif token == "privided":
            token = "provided"
        elif token == "freeflowing":
            token = "free-flowing"
        elif token == "tigher":
            token = "tighter"
        elif token == "flary":
            token = "flare"
        elif token == "capeeffect":
            token = "capeeffect"
        elif token == "onesided":
            token = "one-sided"
        elif token == "compate":
            token = "compare"
        elif token == "offshoulder":
            token = "off-the-shoulder"
        elif token == "strappless":
            token = "strapless"
        elif token == "horizonatally":
            token = "horizontally"
        elif token == "shineir":
            token = "shiner"
        elif token == "tshirt":
            token = "t-shirt"
        elif token == "elgant":
            token = "elegant"
        elif token == "isbrighter":
            token = "brighter"
        elif token == "beltstrap":
            token = "belt"
        elif token == "strappier":
            token = "strapped"
        elif token == "halterneck":
            token = "halterneck"
        elif token == "whtie":
            token = "white"
        elif token == "shoter":
            token = "shorter"
        elif token == "slevves":
            token = "sleeves"
        elif token == "cleeves":
            token = "sleeves"
        elif token == "dotsflower":
            token = "dot-flower"
        elif token == "tiedyed":
            token = "tie-dyed"
        elif token == "moroe":
            token = "more"
        elif token == "flait":
            token = "flat"
        elif token == "leapord":
            token = "leopard"
        elif token == "printes":
            token = "printed"
        elif token == "dlong":
            token = "long"
        elif token == "multple":
            token = "multiple"
        elif token == "sluttier":
            token = "flutter"
        elif token == "shorrter":
            token = "shorter"
        elif token == "stripts":
            token = "stripes"
        elif token == "whire":
            token = "white"
        elif token == "shortsleeves":
            token = "short-sleeved"
        elif token == "simillar":
            token = "similar"
        elif token == "lither":
            token = "lighter"
        elif token == "turquiose":
            token = "turquoise"
        elif token == "sripes":
            token = "stripes"
        elif token == "patern":
            token = "pattern"
        elif token == "coolor":
            token = "colour"
        elif token == "sleves":
            token = "sleeves"
        elif token == "dressm":
            token = "dress"
        elif token == "turqiouse":
            token = "turquoise"
        elif token == "blackred":
            token = "black-red"
        elif token == "ballerinalike":
            token = "ballet"
        elif token == "lengh":
            token = "length"
        elif token == "assymetrical":
            token = "asymmetrical"
        elif token == "shoudler":
            token = "shoulder"
        elif token == "qaurter":
            token = "quarter"
        elif token == "sleevedd":
            token = "sleeved"
        elif token == "fluffly":
            token = "pluffy"
        elif token == "whte":
            token = "white"
        elif token == "pencillike":
            token = "pencil"
        elif token == "midsleeves":
            token = "mid-sleeve"
        elif token == "isblack":
            token = "black"
        elif token == "frnech":
            token = "french"
        elif token == "teired":
            token = "tiered"
        elif token == "shoulderless":
            token = "shoulderless"
        elif token == "polkadots":
            token = "polka-dot"
        elif token == "blus":
            token = "blue"
        elif token == "scoopneck":
            token = "scoop-neck"
        elif token == "longersleeved":
            token = "long-sleeved"
        elif token == "seemlessly":
            token = "seamlessly"
        elif token == "twotone":
            token = "two-tone"
        elif token == "blackdrop":
            token = "backdrop"
        elif token == "onepiece":
            token = "dress"
        elif token == "aptterns":
            token = "patterns"
        elif token == "exier":
            token = "sexier"
        elif token == "andfitted":
            token = "fitted"
        elif token == "winered":
            token = "wintered"
        elif token == "sleevges":
            token = "sleeves"
        elif token == "elagant":
            token = "elegant"
        elif token == "haltertype":
            token = "halterneck"
        elif token == "vneckline":
            token = "v-neck"
        elif token == "vneck":
            token = "v-neck"
        elif token == "earthtoned":
            token = "earth-toned"
        elif token == "colrful":
            token = "colourful"
        elif token == "shortsleeved":
            token = "short-sleeved"
        elif token == "shinr":
            token = "shine"
        elif token == "whote":
            token = "white"
        elif token == "buterfly":
            token = "butterfly"
        elif token == "eparate":
            token = "separate"
        elif token == "dreen":
            token = "green"
        elif token == "flaired":
            token = "flared"
        elif token == "orangegreen":
            token = "orange-green"
        elif token == "beautful":
            token = "beautiful"
        elif token == "fited":
            token = "fitted"
        elif token == "halfsleeves":
            token = "half-sleeved"
        elif token == "frillier":
            token = "frill"
        elif token == "twocolored":
            token = "two-tone"
        elif token == "flowey":
            token = "flowing"
        elif token == "lacier":
            token = "lace"
        elif token == "blackwhite":
            token = "black-white"
        elif token == "disered":
            token = "desired"
        elif token == "contempary":
            token = "contemporary"
        elif token == "stripier":
            token = "striped"
        elif token == "aroung":
            token = "around"
        elif token == "botttom":
            token = "bottom"
        elif token == "midlength":
            token = "mid-length"
        elif token == "prined":
            token = "printed"
        elif token == "wimsy":
            token = "wimpy"
        elif token == "loosefitting":
            token = "loose-fitting"
        elif token == "mosiac":
            token = "mosaic"
        elif token == "sghorter":
            token = "shorter"
        elif token == "offwhite":
            token = "off-white"
        elif token == "sleeveles":
            token = "sleeveless"
        elif token == "brigher":
            token = "brighter"
        elif token == "tiebelt":
            token = "tie-belt"
        elif token == "oldfashioned":
            token = "old-fashioned"
        elif token == "ruffeled":
            token = "ruffled"
        elif token == "shouder":
            token = "shoulder"
        elif token == "feminime":
            token = "feminine"
        elif token == "bototm":
            token = "bottom"
        elif token == "horisontal":
            token = "horizontal"
        elif token == "buttoms":
            token = "buttons"
        elif token == "stribe":
            token = "stripe"
        elif token == "lightercolored":
            token = "lighte-colored"
        elif token == "sking":
            token = "skinny"
        elif token == "strapes":
            token = "stripes"
        elif token == "blackdress":
            token = "black-dress"
        elif token == "stirpes":
            token = "stripes"
        elif token == "crinolin":
            token = "crinoline"
        elif token == "vintag":
            token = "vintage"
        elif token == "satil":
            token = "sail"
        elif token == "buttonup":
            token = "button"
        elif token == "dogfather":
            token = "godfather"
        elif token == "harleydavidson":
            token = "harley-davidson"
        elif token == "flowier":
            token = "flower"
        elif token == "jurk":
            token = "jury"
        elif token == "masculant":
            token = "masculine"
        elif token == "turquose":
            token = "turquoise"
        elif token == "greywhite":
            token = "grey-white"
        elif token == "shoet":
            token = "short"
        elif token == "horsw":
            token = "horse"
        elif token == "sleved":
            token = "sleeved"
        elif token == "neckling":
            token = "neckline"
        elif token == "purplewhite":
            token = "purplewhite"
        elif token == "multipatterned":
            token = "multipatterned"
        elif token == "tshirty":
            token = "T-shirt"
        elif token == "blousey":
            token = "blouse"
        elif token == "offtheshoulder":
            token = "off-the-shoulder"
        elif token == "cown":
            token = "down"
        elif token == "andblue":
            token = "andblue"
        elif token == "sleeeveless":
            token = "sleeveless"
        elif token == "yelllow":
            token = "yellow"
        elif token == "multcolored":
            token = "multicoloured"
        elif token == "colar":
            token = "collar"
        elif token == "yellowgreen":
            token = "yellow-green"
        elif token == "jacketstyle":
            token = "jacketstyle"
        elif token == "awesomed":
            token = "awesome"
        elif token == "burrons":
            token = "buttons"
        elif token == "coloe":
            token = "color"
        elif token == "lavander":
            token = "lavender"
        elif token == "drapped":
            token = "dropped"
        elif token == "tiedye":
            token = "tied"
        elif token == "juvenille":
            token = "juvenile"
        elif token == "camoflaged":
            token = "camouflaged"
        elif token == "iamge":
            token = "image"
        elif token == "reveraling":
            token = "revealing"
        elif token == "whiteblackred":
            token = "whiteblackred"
        elif token == "andsheer":
            token = "landseer"
        elif token == "manlier":
            token = "manly"
        elif token == "sleevess":
            token = "sleeves"
        elif token == "wblack":
            token = "black"
        elif token == "mascuine":
            token = "masculine"
        elif token == "thaats":
            token = "thats"
        elif token == "greenyellow":
            token = "greeny-yellow"
        elif token == "pointedleaves":
            token = "pointedleaves"
        elif token == "freen":
            token = "free"
        elif token == "whitered":
            token = "whitened"
        elif token == "oolor":
            token = "color"
        elif token == "mulitcolored":
            token = "multicoloured"
        elif token == "blackpurpletan":
            token = "blackpurpletan"
        elif token == "flarey":
            token = "flared"
        elif token == "spaggeti":
            token = "spaghetti"
        elif token == "blackblue":
            token = "blackblue"
        elif token == "steeves":
            token = "sleeves"
        elif token == "tyedyed":
            token = "tie-dyed"
        elif token == "grapgic":
            token = "graphic"
        elif token == "grily":
            token = "lily"
        elif token == "unbanded":
            token = "unbranded"
        elif token == "grahic":
            token = "graphic"
        elif token == "orangeblue":
            token = "orangeblue"
        elif token == "musicana":
            token = "musical"
        elif token == "ladycarrying":
            token = "ladycarrying"
        elif token == "scoupe":
            token = "scope"
        elif token == "sparklier":
            token = "sparkler"
        elif token == "withno":
            token = "with"
        elif token == "strapeless":
            token = "strapless"
        elif token == "golfstyle":
            token = "golfstyle"
        elif token == "insterment":
            token = "interment"
        elif token == "moremasculine":
            token = "moremasculine"
        elif token == "seeves":
            token = "serves"
        elif token == "ovalfish":
            token = "coalfish"
        elif token == "featureds":
            token = "features"
        elif token == "yand":
            token = "and"
        elif token == "browngold":
            token = "browngold"
        elif token == "leggins":
            token = "leggings"
        elif token == "printe":
            token = "printed"
        elif token == "nonfitted":
            token = "unfitted"
        elif token == "pantsscarf":
            token = "pantsscarf"
        elif token == "lookinh":
            token = "looking"
        elif token == "sleever":
            token = "sleeve"
        elif token == "stiped":
            token = "striped"
        elif token == "montley":
            token = "motley"
        elif token == "yellowishred":
            token = "yellowishred"
        elif token == "soliid":
            token = "solid"
        elif token == "whits":
            token = "white"
        elif token == "iwth":
            token = "with"
        elif token == "writting":
            token = "writing"
        elif token == "senched":
            token = "reached"
        elif token == "atheltic":
            token = "athletic"
        elif token == "shiry":
            token = "shirt"
        elif token == "printrd":
            token = "printed"
        elif token == "stribes":
            token = "stripes"
        elif token == "recces":
            token = "lace"
        elif token == "waistlength":
            token = "waist-length"
        elif token == "fashonable":
            token = "fashionable"
        elif token == "multiobject":
            token = "multiobject"
        elif token == "revealng":
            token = "revealing"
        elif token == "brownishwhite":
            token = "brownish-white"
        elif token == "gaphic":
            token = "graphic"
        elif token == "sleevse":
            token = "sleeve"
        elif token == "tightd":
            token = "tight"
        elif token == "rulled":
            token = "rolled"
        elif token == "looserfitting":
            token = "loose-fitting"
        elif token == "femininetailored":
            token = "feminine-tailored"
        elif token == "shher":
            token = "sheer"
        elif token == "slimfitted":
            token = "slim-fitted"
        elif token == "plainf":
            token = "plain"
        elif token == "jocker":
            token = "jockey"
        elif token == "prnts":
            token = "prints"
        elif token == "sportsteam":
            token = "sportswear"
        elif token == "mulit":
            token = "multi"
        elif token == "enderman":
            token = "elder-man"
        elif token == "staghe":
            token = "stage"
        elif token == "squatch":
            token = "scratch"
        elif token == "bluewhite":
            token = "blue-white"
        elif token == "plussize":
            token = "plussize"
        elif token == "redbrown":
            token = "red-brown"
        elif token == "tourqoise":
            token = "turquoise"
        elif token == "armygreen":
            token = "armygreen"
        elif token == "waterfally":
            token = "waterfall"
        elif token == "sleaved":
            token = "cleaved"
        elif token == "aleeved":
            token = "sleeved"
        elif token == "preppier":
            token = "premier"
        elif token == "buutton":
            token = "button"
        elif token == "homosapiens":
            token = "homo-sapiens"
        elif token == "grapphics":
            token = "graphics"
        elif token == "whitebrown":
            token = "white-brown"
        elif token == "sheir":
            token = "their"
        elif token == "blackandwhite":
            token = "black-and-white"
        elif token == "uspo":
            token = "spo"
        elif token == "popart":
            token = "pop-art"
        elif token == "lightdark":
            token = "light/dark"
        elif token == "fiting":
            token = "fitting"
        elif token == "pearltrim":
            token = "pearltrim"
        elif token == "darkblue":
            token = "dark-blue"
        elif token == "womanathlete":
            token = "woman-athlete"
        elif token == "bralette":
            token = "palette"
        elif token == "colorfulcar":
            token = "colorful-car"
        elif token == "maam":
            token = "madam"
        elif token == "fittted":
            token = "fitted"
        elif token == "ombr":
            token = "omar"
        elif token == "multicolors":
            token = "multicolour"
        elif token == "deper":
            token = "deeper"
        elif token == "scaryshark":
            token = "scaryshark"
        elif token == "coatofarms":
            token = "coat-of-arms"
        elif token == "rockandroll":
            token = "rock-n-roll"
        elif token == "deniam":
            token = "denial"
        elif token == "racerback":
            token = "paperback"
        elif token == "nect":
            token = "next"
        elif token == "buttondown":
            token = "button-down"
        elif token == "notical":
            token = "notice"
        elif token == "whispier":
            token = "whisper"
        elif token == "firly":
            token = "fairly"
        elif token == "trime":
            token = "trim"
        elif token == "fontks":
            token = "fonts"
        elif token == "rufffles":
            token = "ruffles"
        elif token == "shulders":
            token = "shoulders"
        elif token == "vnck":
            token = "vneck"
        elif token == "turquorse":
            token = "turquoise"
        elif token == "pepplum":
            token = "pepplum"
        elif token == "satinlike":
            token = "satin-like"
        elif token == "jursy":
            token = "jury"
        elif token == "horizontallystriped":
            token = "horizontally-striped"
        elif token == "topstitching":
            token = "top-stitching"
        elif token == "onlay":
            token = "only"
        elif token == "colroed":
            token = "colored"
        elif token == "greygreenkhaki":
            token = "grey-green-khaki"
        elif token == "colorfu":
            token = "colourful"
        elif token == "pantern":
            token = "pattern"
        elif token == "monocolored":
            token = "mono-colored"
        elif token == "shimee":
            token = "shame"
        elif token == "wolfy":
            token = "wolf"
        elif token == "buttonless":
            token = "botton-less"
        elif token == "zepplin":
            token = "zeppelin"
        elif token == "provactive":
            token = "proactive"
        elif token == "carhardt":
            token = "carhartt"
        elif token == "trianglr":
            token = "triangle"
        elif token == "whie":
            token = "while"
        elif token == "flagstyle":
            token = "flag-style"
        elif token == "blueer":
            token = "bluer"
        elif token == "darkre":
            token = "darker"
        elif token == "demim":
            token = "denim"
        elif token == "warz":
            token = "war"
        elif token == "deepercolored":
            token = "deeper-colored"
        elif token == "shitzu":
            token = "shit"
        elif token == "offcenter":
            token = "off-center"
        elif token == "blueblack":
            token = "blue-black"
        elif token == "toung":
            token = "young"
        elif token == "blackgrey":
            token = "black-grey"
        elif token == "slogal":
            token = "slogan"
        elif token == "oranige":
            token = "orange"
        elif token == "sleevesok":
            token = "sleeves"
        elif token == "ballz":
            token = "ball"
        elif token == "polostyle":
            token = "polo-style"
        elif token == "solidcolor":
            token = "solid-color"
        elif token == "tandark":
            token = "standard"
        elif token == "purplel":
            token = "purple"
        elif token == "constrast":
            token = "contrast"
        elif token == "etchedstyle":
            token = "etched-style"
        elif token == "bullethole":
            token = "bullet-hole"
        elif token == "hotsauce":
            token = "hot-sauce"
        elif token == "whiteblack":
            token = "white-black"
        elif token == "yelloworangered":
            token = "yellow-orangered"
        elif token == "xmen":
            token = "men"
        elif token == "wordier":
            token = "worrier"
        elif token == "colros":
            token = "colors"
        elif token == "tcpip":
            token = "tcp/ip"
        elif token == "grayishwhite":
            token = "greyish-white"
        elif token == "tshty":
            token = "they"
        elif token == "greyblackwhite":
            token = "grey-black-white"
        elif token == "grar":
            token = "gear"
        elif token == "bluegray":
            token = "blue-grey"
        elif token == "buttions":
            token = "buttons"
        elif token == "slimfit":
            token = "limit"
        elif token == "jimis":
            token = "jimi"
        elif token == "lablels":
            token = "labels"
        elif token == "liony":
            token = "lion"
        elif token == "redgrey":
            token = "regret"
        elif token == "redwhite":
            token = "red-write"
        elif token == "monocromatic":
            token = "mono-chromatic"
        elif token == "curcular":
            token = "circular"
        elif token == "sorange":
            token = "strange"
        elif token == "yshirt":
            token = "shirt"
        elif token == "clorful":
            token = "colourful"
        elif token == "purpleblue":
            token = "purple-blue"
        elif token == "whitegray":
            token = "white-grey"
        elif token == "blueorange":
            token = "blue-orange"
        elif token == "fullbuttoned":
            token = "full-buttoned"
        elif token == "lightcolored":
            token = "light-coloured"
        elif token == "colorul":
            token = "colour"
        elif token == "undersleeve":
            token = "under-sleeve"
        elif token == "handmark":
            token = "landmark"
        elif token == "riny":
            token = "ring"
        elif token == "gangsterish":
            token = "gangsterism"
        elif token == "graphihc":
            token = "graphic"
        elif token == "deesign":
            token = "design"
        elif token == "crks":
            token = "corks"
        elif token == "guntheme":
            token = "gun-theme"
        elif token == "greyblue":
            token = "grey-blue"
        elif token == "dinosaurus":
            token = "dinosaurs"
        elif token == "animalted":
            token = "animated"
        elif token == "multipack":
            token = "multipack"
        elif token == "silhouettelike":
            token = "silhouette-like"
        elif token == "bluebrown":
            token = "blue-brown"
        elif token == "redstriped":
            token = "red-striped"
        elif token == "monly":
            token = "only"
        elif token == "gitting":
            token = "getting"
        elif token == "frong":
            token = "wrong"
        elif token == "sexuak":
            token = "sexual"
        elif token == "buttonfront":
            token = "button-front"
        elif token == "quotel":
            token = "quoted"
        elif token == "greenbrown":
            token = "grey-brown"
        elif token == "squatching":
            token = "scratching"
        elif token == "foundland":
            token = "foundling"
        elif token == "startrack":
            token = "startrack"
        elif token == "slinkier":
            token = "linker"
        elif token == "promienent":
            token = "prominent"
        elif token == "graphica":
            token = "graphics"
        elif token == "pistolwhip":
            token = "pistol-whip"
        elif token == "patiriotic":
            token = "patriotic"
        elif token == "orgasam":
            token = "orgasm"
        elif token == "onit":
            token = "unit"
        elif token == "pastelcolored":
            token = "pastel-coloured"
        elif token == "enimem":
            token = "enamel"
        elif token == "humanfocused":
            token = "human-focused"
        elif token == "eveningfocused":
            token = "evening-focused"
        elif token == "rockos":
            token = "rocks"
        elif token == "redblue":
            token = "red-blue"
        elif token == "vacationy":
            token = "vacation"
        elif token == "sprtier":
            token = "sportier"
        elif token == "clevage":
            token = "cleavage"
        elif token == "plaided":
            token = "plaited"
        elif token == "masciline":
            token = "masculine"
        elif token == "juvemile":
            token = "juvenile"
        elif token == "pladi":
            token = "plaid"  # here
        elif token == "semistriped":
            token = "striped"
        elif token == "oclor":
            token = "color"
        elif token == "hsorter":
            token = "shorter"
        elif token == "sleevels":
            token = "sleeves"
        elif token == "bleted":
            token = "belted"
        elif token == "pencel":
            token = "pencil"
        elif token == "releavant":
            token = "relevant"
        elif token == "sraps":
            token = "straps"
        elif token == "flwoing":
            token = "flowing"
        elif token == "anklelength":
            token = "ankle-length"
        elif token == "onetoned":
            token = "one-tone"
        elif token == "multiclored":
            token = "multicoloured"
        elif token == "armlength":
            token = "arm-length"
        elif token == "sflowing":
            token = "flowing"
        elif token == "dtraps":
            token = "straps"
        elif token == "powtie":
            token = "bowtie"
        elif token == "spaghettistrapped":
            token = "spaghetti-strap"
        elif token == "onecolored":
            token = "one-tone"
        elif token == "whiye":
            token = "white"
        elif token == "leporad":
            token = "leopard"
        elif token == "redorange":
            token = "red-orange"
        elif token == "stirped":
            token = "striped"
        elif token == "revelaing":
            token = "revealing"
        elif token == "blackpink":
            token = "black-pink"
        elif token == "coloed":
            token = "coloured"
        elif token == "tellow":
            token = "yellow"
        elif token == "silverlike":
            token = "silver"
        elif token == "pattetn":
            token = "pattern"
        elif token == "pastal":
            token = "pastel"
        elif token == "vlack":
            token = "black"
        elif token == "sholuders":
            token = "shoulders"
        elif token == "struped":
            token = "striped"
        elif token == "onetone":
            token = "one-tone"
        elif token == "onearmed":
            token = "one-armed"
        elif token == "romperlike":
            token = "romper"
        elif token == "kehole":
            token = "keyhole"
        elif token == "highneck":
            token = "turtleneck"
        elif token == "capeless":
            token = "capless"
        elif token == "romber":
            token = "romper"
        elif token == "dot-flower":
            token = "dot-flower"
        elif token == "shint":
            token = "shine"
        elif token == "pluffy":
            token = "fluffy"
        elif token == "mid-sleeve":
            token = "mid-sleeve"
        elif token == "orange-green":
            token = "orange-green"
        elif token == "half-sleeved":
            token = "half-sleeved"
        elif token == "cheetahpattern":
            token = "leopard"
        elif token == "lighte-colored":
            token = "light-coloured"
        elif token == "black-dress":
            token = "black"
        elif token == "belr":
            token = "belt"
        elif token == "fittiing":
            token = "fitting"
        elif token == "colordul":
            token = "colourful"
        elif token == "yeloow":
            token = "yellow"
        elif token == "clolorful":
            token = "colourful"
        elif token == "plussized":
            token = "plussized"
        elif token == "pinkgray":
            token = "pinkgrey"
        elif token == "satinier":
            token = "statin"
        elif token == "comparisoin":
            token = "comparison"
        elif token == "colorfol":
            token = "colourful"
        elif token == "stapless":
            token = "strapless"
        elif token == "shortrer":
            token = "shorter"
        elif token == "geormetric":
            token = "geometric"
        elif token == "pattertn":
            token = "pattern"
        elif token == "desiered":
            token = "desired"
        elif token == "probuct":
            token = "product"
        elif token == "colvering":
            token = "covering"
        elif token == "midsleeved":
            token = "mid-sleeve"
        elif token == "necklne":
            token = "neckline"
        elif token == "ligter":
            token = "lighter"
        elif token == "flullier":
            token = "fuller"
        elif token == "shinky":
            token = "shrink"
        elif token == "ayered":
            token = "layered"
        elif token == "strip0es":
            token = "stripes"
        elif token == "strapts":
            token = "straps"
        elif token == "nexk":
            token = "next"
        elif token == "muticolored":
            token = "multicoloured"
        elif token == "skeevs":
            token = "sleeves"
        elif token == "datk":
            token = "dark"
        elif token == "dotten":
            token = "dotted"
        elif token == "layereduh":
            token = "layered"
        elif token == "allwhite":
            token = "white"
        elif token == "darket":
            token = "darker"
        elif token == "masucline":
            token = "masculine"
        elif token == "camolagued":
            token = "camouflage"
        elif token == "geometrric":
            token = "geometric"
        elif token == "addc":
            token = "add"
        elif token == "abutton":
            token = "button"
        elif token == "redwhiteand":
            token = "redwhite"
        elif token == "butoned":
            token = "buttoned"
        elif token == "centeredskull":
            token = "centeredskull"
        elif token == "graygreen":
            token = "greygreen"
        elif token == "tshirtlike":
            token = "t-shirt"
        elif token == "7pack":
            token = "pack"
        elif token == "unprofessionals":
            token = "unprofessional"
        elif token == "casaual":
            token = "casual"
        elif token == "flannerl":
            token = "flannel"
        elif token == "symmmetrical":
            token = "symmetrical"
        elif token == "greend":
            token = "green"
        elif token == "morewhite":
            token = "white"
        elif token == "brightercolored":
            token = "brighter"
        elif token == "animalprinted":
            token = "animal"
        elif token == "imagew":
            token = "image"
        elif token == "letterin":
            token = "lettering"
        elif token == "botton-less":
            token = "bottonless"
        elif token == "flag-style":
            token = "flag-style"
        elif token == "deeper-colored":
            token = "deeper-colored"
        elif token == "batsignal":
            token = "basinal"
        elif token == "etched-style":
            token = "etched-style"
        elif token == "davincis":
            token = "davincis"
        elif token == "yellow-orangered":
            token = "yellow-orange"
        elif token == "gun-theme":
            token = "gun-theme"
        elif token == "silhouette-like":
            token = "silhouette-like"
        elif token == "footballfish":
            token = "footballfish"
        elif token == "startrack":
            token = "startrack"
        elif token == "human-focused":
            token = "human-focused"
        elif token == "evening-focused":
            token = "evening-focused"
        elif token == "alighter":
            token = "lighter"
        elif token == "agray":
            token = "grey"
        elif token == "tshirtstyle":
            token = "t-shirt"
        elif token == "coloorful":
            token = "colourful"
        elif token == "selfimage":
            token = "self-image"
        elif token == "slleves":
            token = "sleeves"
        elif token == "bodyuilding":
            token = "bodybuilding"
        elif token == "brownishorange":
            token = "brownishorange"
        elif token == "adulty":
            token = "adult"
        elif token == "poiinted":
            token = "pointed"
        elif token == "mathrelated":
            token = "math-related"
        elif token == "graphiic":
            token = "graphic"
        elif token == "strioe":
            token = "stripe"
        elif token == "fourletter":
            token = "four-letter"
        elif token == "darkercolored":
            token = "darker"
        elif token == "coloered":
            token = "coloured"
        elif token == "grend":
            token = "trend"
        elif token == "mutlicolored":
            token = "multicoloured"
        elif token == "samell":
            token = "small"
        elif token == "hwhite":
            token = "white"
        elif token == "coloureed":
            token = "coloured"
        elif token == "symblol":
            token = "symbol"
        elif token == "islighter":
            token = "lighter"
        elif token == "bluetshirt":
            token = "blue"
        elif token == "isstripped":
            token = "striped"
        elif token == "oclorful":
            token = "colourful"
        elif token == "halfzipper":
            token = "zipper"
        elif token == "yellowredgreen":
            token = "yellowgreen"
        elif token == "yelloworange":
            token = "yelloworange"
        elif token == "detaill":
            token = "details"
        elif token == "lacylooking":
            token = "lady"
        elif token == "sleeeves":
            token = "sleeves"
        elif token == "clolor":
            token = "colour"
        elif token == "staggeti":
            token = "spaghetti"
        elif token == "lxve":
            token = "love"
        elif token == "colorflul":
            token = "colourful"
        elif token == "pinkgreen":
            token = "pinkgreen"
        elif token == "vshaped":
            token = "v-neck"
        elif token == "sleveeves":
            token = "sleeves"
        elif token == "keyholed":
            token = "keyhole"
        elif token == "lgoo":
            token = "logo"
        elif token == "sleefes":
            token = "sleeves"
        elif token == "peasantlike":
            token = "peasant"
        elif token == "ablouse":
            token = "blouse"
        elif token == "wtriangle":
            token = "triangle"
        elif token == "flimsyblack":
            token = "flimsyblack"
        elif token == "qriting":
            token = "writing"
        elif token == "strpes":
            token = "stripes"
        elif token == "graphci":
            token = "graphic"
        elif token == "lnger":
            token = "longer"
        elif token == "ahtletic":
            token = "athletic"
        elif token == "sphisticated":
            token = "sophisticated"
        elif token == "shirrt":
            token = "shirt"
        elif token == "footballshirt":
            token = "footballshirt"
        elif token == "spnaiels":
            token = "spaniels"
        elif token == "froont":
            token = "front"
        elif token == "shordter":
            token = "shorter"
        elif token == "purplewhite":
            token = "purplewhite"
        elif token == "multipatterned":
            token = "multipatterned"
        elif token == "equalsize":
            token = "equalize"
        elif token == "andblue":
            token = "blue"
        elif token == "jacketstyle":
            token = "jacket"
        elif token == "equatoins":
            token = "equations"
        elif token == "merlottes":
            token = "merlottes"
        elif token == "englishonly":
            token = "english"
        elif token == "whiteblackred":
            token = "whiteblackred"
        elif token == "repeatedsentence":
            token = "sentence"
        elif token == "ladycarrying":
            token = "lady"
        elif token == "golfstyle":
            token = "golf"
        elif token == "multiobject":
            token = "multiobject"
        elif token == "bhudda":
            token = "buddha"
        elif token == "feminine-tailored":
            token = "feminine-tailored"
        elif token == "slim-fitted":
            token = "slim-fitted"
        elif token == "armygreen":
            token = "camouflage"
        elif token == "white-brown":
            token = "white-brown"
        elif token == "greaan":
            token = "green"
        elif token == "woman-athlete":
            token = "woman-athlete"
        elif token == "colorful-car":
            token = "colorful-car"
        elif token == "scaryshark":
            token = "scaryshark"
        elif token == "horizontally-striped":
            token = "horizontally-striped"
        elif token == "taank":
            token = "tank"
        elif token == "femanen":
            token = "female"
        elif token == "sleevelass":
            token = "sleeveless"
        elif token == "nackline":
            token = "neckline"
        elif token == "nonbutton":
            token = "no-button"
        elif token == "sleevedesign":
            token = "sleeve"
        elif token == "midsleeve":
            token = "mid-sleeve"
        elif token == "garphic":
            token = "graphic"
        elif token == "oneshoulder":
            token = "one-shoulder"
        elif token == "inspirsed":
            token = "inspired"
        elif token == "humanfigure":
            token = "humanfigure"
        elif token == "peachcolored":
            token = "peach-coloured"
        elif token == "vnecl":
            token = "v-neck"
        elif token == "winebottle":
            token = "winebottle"
        elif token == "thinnder":
            token = "thinner"
        elif token == "dlirty":
            token = "dirty"
        elif token == "necklink":
            token = "neckline"
        elif token == "dresslength":
            token = "dresslength"
        elif token == "turtlneck":
            token = "turtleneck"
        elif token == "offsoulder":
            token = "off-the-shoulder"
        elif token == "revleaing":
            token = "revealing"
        elif token == "paatern":
            token = "pattern"
        elif token == "revealingg":
            token = "revealing"
        elif token == "burgancy":
            token = "burgundy"
        elif token == "solidgray":
            token = "solidgrey"
        elif token == "comforty":
            token = "comfort"
        elif token == "speggati":
            token = "spaghetti"
        elif token == "lntament":
            token = "lnstrument"
        elif token == "vingate":
            token = "vintage"
        elif token == "singleshouldered":
            token = "one-shoulder"
        elif token == "colot":
            token = "color"
        elif token == "elss":
            token = "less"
        elif token == "shorteer":
            token = "shorter"
        elif token == "organe":
            token = "orange"
        elif token == "coldshoulder":
            token = "cold shoulder"
        elif token == "slimfitted":
            token = "slim fit"
        elif token == "multiobject":
            token = "multi object"
        elif token == "horizontallystriped":
            token = "horizontal stripe"
        elif token == "blueblack":
            token = "blue black"
        elif token == "whiteblackred":
            token = "white black red"
        elif token == "doted":
            token = "dotted"
        elif token == "asymmetrically":
            token = "asymmetrically"
        elif token == "ladycarrying":
            token = "lady carrying"
        elif token == "dotflower":
            token = "dot flower"
        elif token == "highwaisted":
            token = "high waisted"
        elif token == "footballfish":
            token = "football fish"
        elif token == "blackandwhite":
            token = "black and white"
        elif token == "leafed":
            token = "leaf"
        elif token == "greypink":
            token = "grey pink"
        elif token == "purplewhite":
            token = "purple white"
        elif token == "sleevesshoulder":
            token = "sleeveless shoulder"
        elif token == "kneelength":
            token = "knee length"
        elif token == "undersleeve":
            token = "under sleeve"
        elif token == "yelloworangered":
            token = "yellow orange red"
        elif token == "redblue":
            token = "red blue"
        elif token == "hasmodel":
            token = "has model"
        elif token == "pointedleaves":
            token = "pointed leaves"
        elif token == "afican":
            token = "african"
        elif token == "jacketstyle":
            token = "jacket style"
        elif token == "mustach":
            token = "mustach"
        elif token == "yellower":
            token = "yellower"
        elif token == "multipatterned":
            token = "multi patterned"
        elif token == "blueorange":
            token = "blue orange"
        elif token == "scaryshark":
            token = "scary shark"
        elif token == "whitegrey":
            token = "white grey"
        elif token == "blackwhite":
            token = "black white"
        elif token == "deigns":
            token = "design"
        elif token == "orangegreen":
            token = "orange green"
        elif token == "bluegrey":
            token = "blue grey"
        elif token == "lightecolored":
            token = "light color"
        elif token == "pattered":
            token = "patterned"

        if not prev_token == token:
            print(f"Correct token {prev_token} --> {token}") if __VERBOSE__ else ""

        return token
