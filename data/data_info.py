
def get_data_info(args):
    if args.dataset == "PACS":
        args.Domain_ID = ['art_painting', 'sketch', 'photo', 'cartoon']
        args.classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        args.n_classes = 7
        args.n_domain = 4
    elif args.dataset == "VLCS":
        args.Domain_ID = ["LabelMe", "SUN", "VOC", "Caltech"]
        args.classes = ["bird", "car", "chair", "dog", "person"]
        args.n_classes = 5
        args.n_domain = 4
    elif args.dataset == "Terra":
        args.Domain_ID = ["location_100", "location_38", "location_43", "location_46"]
        args.classes = ["bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit", "raccoon", "squirrel"]
        args.n_classes = 10
        args.n_domain = 4
    elif args.dataset == "Officehome":
        args.Domain_ID = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.classes = ["Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket", "Calculator",
                        "Calendar", "Candles", "Chair", "Clipboards", "Computer", "Couch", "Curtains", "Desk_Lamp",
                        "Drill", "Eraser", "Exit_Sign", "Fan", "File_Cabinet", "Flipflops", "Flowers", "Folder", "Fork",
                        "Glasses", "Hammer", "Helmet", "Kettle", "Keyboard", "Knives", "Lamp_Shade", "Laptop", "Marker",
                        "Monitor", "Mop", "Mouse", "Mug", "Notebook", "Oven", "Pan", "Paper_Clip", "Pen", "Pencil",
                        "Postit_Notes", "Printer", "Push_Pin", "Radio", "Refrigerator", "Ruler", "Scissors",
                        "Screwdriver", "Shelf", "Sink", "Sneakers", "Soda", "Speaker", "Spoon", "Table", "Telephone",
                        "ToothBrush", "Toys", "Trash_Can", "TV", "Webcam"]
        args.n_classes = 65
        args.n_domain = 4
    elif args.dataset == "Office-31":
        args.Domain_ID = ['amazon', 'dslr', 'webcam']
        args.classes = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle", "calculator", "desk_chair",
                        "desk_lamp", "desktop_computer", "file_cabinet", "headphones", "keyboard", "laptop_computer",
                        "letter_tray", "mobile_phone", "monitor", "mouse", "mug", "paper_notebook", "pen", "phone",
                        "printer", "projector", "punchers", "ring_binder", "ruler", "scissors", "speaker", "stapler",
                        "tape_dispenser", "trash_can"]
        args.n_classes = 31
        args.n_domain = 3
    elif args.dataset == "VisDA":
        args.Domain_ID = ['synthetic', 'real']
        args.classes = ["aeroplane", "bicycle", "bus", "car", "horse", "knife", "motorcycle", "person", "plant",
                        "skateboard", "train", "truck"]
        args.n_classes = 12
        args.n_domain = 2
    elif args.dataset == "DomainNet-126":
        args.Domain_ID = ["clipart", "painting", "real", "sketch"]
        args.classes = ["aircraft_carrier", "alarm_clock", "ant", "anvil", "asparagus", "axe",
                        "banana", "basket", "bathtub", "bear", "bee", "bird", "blackberry",
                        "blueberry", "bottlecap", "broccoli", "bus", "butterfly", "cactus",
                        "cake", "calculator", "camel", "camera", "candle", "cannon", "canoe",
                        "carrot", "castle", "cat", "ceiling_fan", "cello", "cell_phone", "chair",
                        "chandelier", "coffee_cup", "compass", "computer", "cow", "crab",
                        "crocodile", "cruise_ship", "dog", "dolphin", "dragon", "drums", "duck",
                        "dumbbell", "elephant", "eyeglasses", "feather", "fence", "fish",
                        "flamingo", "flower", "foot", "fork", "frog", "giraffe", "goatee",
                        "grapes", "guitar", "hammer", "helicopter", "helmet", "horse", "kangaroo",
                        "lantern", "laptop", "leaf", "lion", "lipstick", "lobster", "microphone",
                        "monkey", "mosquito", "mouse", "mug", "mushroom", "onion", "panda",
                        "peanut", "pear", "peas", "pencil", "penguin", "pig", "pillow",
                        "pineapple", "potato", "power_outlet", "purse", "rabbit", "raccoon",
                        "rhinoceros", "rifle", "saxophone", "screwdriver", "sea_turtle", "see_saw",
                        "sheep", "shoe", "skateboard", "snake", "speedboat", "spider", "squirrel",
                        "strawberry", "streetlight", "string_bean", "submarine", "swan", "table",
                        "teapot", "teddy-bear", "television", "The_Eiffel_Tower",
                        "The_Great_Wall_of_China", "tiger", "toe", "train", "truck", "umbrella",
                        "vase", "watermelon", "whale", "zebra"]
        args.n_classes = 126
        args.n_domain = 4
    else:
        raise NotImplementedError
