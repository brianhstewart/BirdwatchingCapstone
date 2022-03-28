import io
import sys
import tensorflow as tf
import PySimpleGUI as psg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



# from tensorflow import keras
from PIL import Image

import os


def run_prediction(file_name, model, class_names):
    """
    Has the model (.h5 file) predict the species of bird in the given image.
    :param file_name: the image being scored
    :param model: the model making the decision
    :param class_names: the list of class names we made from the test data
    :return: species is the species prediction, confidence is the percent of certainty the model has
    """
    image = tf.keras.preprocessing.image.load_img(file_name, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_batch = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_batch)
    species = class_names[np.argmax(prediction[0])]
    confidence = max(prediction[0])
    return species, confidence, prediction[0]


def temp_testing_function(model, class_names):
    """
    This function is meant to be called as a way to find patterns in the model's mistakes
    and to double check that it is correctly guessing for most images.  It is not meant to be
    called outside of testing, but i found writing and using the function to be an educational
    exercise since this is my first ML project.
    :param model: the model being run through the test data
    :param class_names: the list of class names for this set
    :return: none.  just prints data to console for the programmer to view.
    """
    correct = 0
    incorrect = 0
    file_names = []
    count = 0
    for folder in os.listdir("test"):
        for f in os.listdir("test/" + folder):
            if os.path.isfile(os.path.join(("test/" + folder), f)) and f.lower().endswith(".jpg"):
                file_names.append("test/{}/{}".format(folder, f))
    for pic in file_names:
        guess, conf, ignore = run_prediction(pic, model, class_names)
        count += 1
        if count % 100 == 0:
            print("{}/1750".format(count))
        if conf < .9:
            print("{} at {}".format(conf * 100, pic))
        if guess in pic:
            correct += 1
        else:
            incorrect += 1
            print("Incorrect at {}".format(pic))
            if conf > 9:
                print("INVESTIGATE? "
                      "-----------------------------------------------------------------------------------------------")
    print("FINAL: \n\tCorrect: {}\n\tIncorrect: {}".format(correct, incorrect))


def draw_figure(canvas, figure):
    """
    Draws the figure on to the given Canvas object
    :param canvas: Canvas object for drawing on to
    :param figure: Graph to be drawn
    :return: reference to drawn figure
    """
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def clear_figure(figure_agg):
    """
    erases the drawn figure from the GUI
    :param figure_agg: the figure/graph to be cleared
    """
    figure_agg.get_tk_widget().forget()
    plt.close('all')


def pie_chart(predictions, class_names):
    """
    displays a pie chart of the predictions given by percentage of confidence.

    predictions from the run_prediction function are expected.
    :param predictions: the array of predictions returned by run_predictions
    :param class_names: array of class names in the order the AI uses.
    :return: No return.  Displays a Pie Chart as popup.
    """
    names = []
    values = []
    count = -1
    for value in predictions:
        count += 1
        if value > 0.001:
            names.append(class_names[count])
            values.append(value)
    pie_chart = plt.pie(values)
    plt.legend(labels=names, title="Possible Species:", loc="best", )
    plt.show()


def accuracy_chart():
    """
    creates the bar chart that compares the accuracy of the three model types
    :return: bar chart with correct statistics
    """
    plt.clf()
    values = (94.3, 87.1, 85)
    index = np.arange(len(values))
    bar_chart = plt.bar(index, values, 0.4, color=("blue"))
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy % in Tests")
    plt.xticks(index, ("Xception", "AlexNet", "ResNet"))
    plt.yticks(np.arange(0, 101, 10))
    return plt.gcf()
    # plt.legend((bar_chart[0],), (""))


def training_time_chart():
    """
    creates the bar chart that compares the training times of the three model types
    :return: bar chart with correct statistics
    """
    values = (4.0, 5 / 6, 3.8)
    index = np.arange(len(values))
    bar_chart = plt.bar(index, values, 0.4, color=("red"))
    plt.title("Test Time Comparison")
    plt.ylabel("Minutes per Training Round")
    plt.xticks(index, ("Xception", "AlexNet", "ResNet"))
    plt.yticks(np.arange(0, 5.1, 0.5))
    return plt.gcf()


def epochs_chart():
    """
    creates the bar chart that compares the epochs needed for each of the three model types
    :return: bar chart with correct statistics
    """
    values = (124, 65, 84)
    index = np.arange(len(values))
    bar_chart = plt.bar(index, values, 0.4, color=("green"))
    plt.title("Training Rounds Comparison")
    plt.ylabel("Training Rounds till Peak Accuracy")
    plt.xticks(index, ("Xception", "AlexNet", "ResNet"))
    plt.yticks(np.arange(0, 126, 25))
    return plt.gcf()


def info_button_trigger(return_window):
    """
    Creates a window that displays info about the three models we used in this program.  Each program has its
    peak accuracy alongside the amount of training rounds it took to achieve that accuracy, and the approximate
    training time each round took on my machine.
    :param return_window: the window we need to return to after the user has read this information
    :return: does not return anything
    """
    figure = accuracy_chart()
    fig_x, fig_y, fig_w, fig_h = figure.bbox.bounds
    graph_counter = 0

    graph_column = [
        [
            psg.Canvas(size=(fig_w, fig_h), key="-CANVAS-")
        ],
        [
            psg.Button("Show Next Graph", key="-CONTINUE-")
        ]
    ]
    text_column = [
        [
            psg.Text(
                "The Xception model we trained across 125 iterations found it's peak accuracy on the 124th epoch\n"
                "\tand took about 4 minutes of training per epoch on my machine.  This model has\n"
                "\t94.3% accuracy during tests, making it our first model.")
        ],
        [
            psg.HSeparator()
        ],
        [
            psg.Text(
                "The AlexNet model we trained across 125 iterations found it's peak accuracy on the 65th epoch\n"
                "\tand took about 50 seconds of training per epoch on my machine.  This model has\n"
                "\t87.1% accuracy during tests, making it our second model.")
        ],
        [
            psg.HSeparator()
        ],
        [
            psg.Text(
                "The ResNet model we trained across 125 iterations found it's peak accuracy on the 84th epoch\n"
                "\tand took a little less than 4 minutes of training per epoch on my machine.  This model has\n"
                "\t85% accuracy during tests, making it our third model.")
        ],
        [
            psg.HSeparator()
        ],
        [
            psg.Button("Return to Homepage", key="-RETURN-")
        ]
    ]

    display = [[[psg.Column(graph_column)], [psg.HSeparator()], [psg.Column(text_column)]]]
    window = psg.Window("Model Info Page", display, finalize=True)
    fig_photo = draw_figure(window['-CANVAS-'].TKCanvas, figure)
    return_var = False
    while True:
        event, values = window.read()
        if event == "Exit" or event == psg.WIN_CLOSED:
            return_window.close()
            window.close()
            break
        if event == "-RETURN-":
            return_var = True
            clear_figure(fig_photo)
            window.close()
            return_window.UnHide()
            break
            # os.system("pause")
        elif event == "-CONTINUE-":
            clear_figure(fig_photo)
            graph_counter += 1
            graph_counter %= 3
            if graph_counter == 0:
                figure = accuracy_chart()
            if graph_counter == 1:
                figure = training_time_chart()
            if graph_counter == 2:
                figure = epochs_chart()
            fig_photo = draw_figure(window['-CANVAS-'].TKCanvas, figure)
            window.refresh()
    clear_figure(fig_photo)
    window.close()
    if return_var:
        return_window.UnHide()
        return_var = False


def main_page_actions(class_names, Xception_model, AlexNet_model, ResNet_model):
    """
    This function creates the main window and manages its functions.  It allows the user to select a folder with
    photos to analyze, and sends the calls to get the model predictions when the user selects a photo.  The program
    currently only takes .jpg files since that is what the models were trained on.  It also has a button that sends the
    user to a window displaying information about the models.
    :param class_names: a list of class names in the order the models know them
    :param Xception_model: the loaded model with the Xception framework
    :param AlexNet_model: the loaded model with the AlexNet framework
    :param ResNet_model: the loaded model with the ResNet framework
    :return: Does not return any values
    """
    file_list_column = [
        [
            psg.Text("Image Folder"),
            psg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            psg.FolderBrowse(),
        ],
        [
            psg.Listbox(
                values=[], enable_events=True, size=(40, 10), key="-FILE LIST-"
            )
        ],
    ]
    image_viewer_column = [
        [psg.Text("Choose an image from list on left:")],
        [psg.Text(size=(50, 1), key="-TOUT-")],
        [psg.Image(key="-IMAGE-")],
    ]
    xcr_readout = [
        [psg.Text("Xception Results:   ")],
        [psg.Text(size=(70, 1), key="-XCR-")],
    ]
    x_button = [[psg.Button("More Detail: Xception", key="-XCONF-")]]
    anr_readout = [
        [psg.Text("AlexNet Results:    ")],
        [psg.Text(size=(70, 1), key="-ANR-")],
    ]
    a_button = [[psg.Button("More Detail: AlexNet", key="-ACONF-")]]
    rnr_readout = [
        [psg.Text("ResNet Results:     ")],
        [psg.Text(size=(70, 1), key="-RNR-")],
    ]
    r_button = [[psg.Button("More Detail: ResNet", key="-RCONF-")]]
    info_button = [
        [psg.Button("AI Model Info", key="-INFO-")]
    ]
    group = [
        [psg.Column(file_list_column), psg.VSeparator(), psg.Column(image_viewer_column)],
        [psg.HSeparator()],
        [psg.Column(xcr_readout), psg.Column(x_button)],
        [psg.Column(anr_readout), psg.Column(a_button)],
        [psg.Column(rnr_readout), psg.Column(r_button)],
        [psg.Column(info_button)]
    ]
    # ta1 = [[psg.Text("(94.3% Tested Accuracy)")]]
    # ta2 = [[psg.Text("(87.1% Tested Accuracy)")]]
    # ta3 = [[psg.Text("(85.0% Tested Accuracy)")]]
    layout = [
        [
            psg.Column(group)
        ]
    ]
    window = psg.Window("Bird JPG Analyzer", layout, finalize=True)
    while True:
        event, values = window.read()
        if event == "Exit" or event == psg.WIN_CLOSED:
            break
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                file_list = os.listdir(folder)
            except:
                file_list = []

            file_names = []
            for f in file_list:
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(".jpg"):
                    file_names.append(f)
            window["-FILE LIST-"].update(file_names)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                image = Image.open(filename)
                image.thumbnail((450, 450))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-TOUT-"].update(filename)
                window["-IMAGE-"].update(data=bio.getvalue())
                s1, c1, for_pie_chart = run_prediction(filename, Xception_model, class_names)
                xcr = "The species is believed to be {species} with {conf:.1f}% certainty" \
                    .format(species=s1, conf=(c1 * 100))
                s2, c2, ignore = run_prediction(filename, AlexNet_model, class_names)
                anr = "The species is believed to be {species} with {conf:.1f}% certainty" \
                    .format(species=s2, conf=(c2 * 100))
                s3, c3, ignore = run_prediction(filename, ResNet_model, class_names)
                rnr = "The species is believed to be {species} with {conf:.1f}% certainty" \
                    .format(species=s3, conf=(c3 * 100))
                window["-XCR-"].update(xcr)
                window["-ANR-"].update(anr)
                window["-RNR-"].update(rnr)
            except:
                pass
        elif event == "-INFO-":
            window.Hide()
            info_button_trigger(window)
        elif event == "-XCONF-":
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                ignore, ignore, prediction = run_prediction(filename, Xception_model, class_names)
                pie_chart(prediction, class_names)
            except:
                text = "Sorry, please select a valid file before using this function."
                window["-XCR-"].update(text)
                pass
        elif event == "-ACONF-":
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                ignore, ignore, prediction = run_prediction(filename, AlexNet_model, class_names)
                pie_chart(prediction, class_names)
            except:
                text = "Sorry, please select a valid file before using this function."
                window["-ANR-"].update(text)
                pass
        elif event == "-RCONF-":
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                ignore, ignore, prediction = run_prediction(filename, ResNet_model, class_names)
                pie_chart(prediction, class_names)
            except:
                text = "Sorry, please select a valid file before using this function."
                window["-RNR-"].update(text)
                pass
    window.close()


if __name__ == '__main__':
    # determine if application is a script file or frozen exe
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    elif __file__:
        application_path = os.path.dirname(__file__)

    Xception_model = os.path.join(application_path, "Best_Xception.h5")
    AlexNet_model = os.path.join(application_path, "Best_AlexNet.h5")
    ResNet_model = os.path.join(application_path, "Best_ResNet.h5")

    Xception_model = tf.keras.models.load_model(Xception_model)
    AlexNet_model = tf.keras.models.load_model(AlexNet_model)
    ResNet_model = tf.keras.models.load_model(ResNet_model)
    class_names = ['ABBOTTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL', 'AFRICAN CROWNED CRANE',
                   'AFRICAN EMERALD CUCKOO', 'AFRICAN FIREFINCH', 'AFRICAN OYSTER CATCHER', 'ALBATROSS',
                   'ALBERTS TOWHEE', 'ALEXANDRINE PARAKEET', 'ALPINE CHOUGH', 'ALTAMIRA YELLOWTHROAT',
                   'AMERICAN AVOCET', 'AMERICAN BITTERN', 'AMERICAN COOT', 'AMERICAN GOLDFINCH', 'AMERICAN KESTREL',
                   'AMERICAN PIPIT', 'AMERICAN REDSTART', 'AMETHYST WOODSTAR', 'ANDEAN GOOSE', 'ANDEAN LAPWING',
                   'ANDEAN SISKIN', 'ANHINGA', 'ANIANIAU', 'ANNAS HUMMINGBIRD', 'ANTBIRD', 'ANTILLEAN EUPHONIA',
                   'APAPANE', 'APOSTLEBIRD', 'ARARIPE MANAKIN', 'ASHY THRUSHBIRD', 'ASIAN CRESTED IBIS', 'AZURE JAY',
                   'AZURE TANAGER', 'AZURE TIT', 'BAIKAL TEAL', 'BALD EAGLE', 'BALD IBIS', 'BALI STARLING',
                   'BALTIMORE ORIOLE', 'BANANAQUIT', 'BAND TAILED GUAN', 'BANDED BROADBILL', 'BANDED PITA',
                   'BANDED STILT', 'BAR-TAILED GODWIT', 'BARN OWL', 'BARN SWALLOW', 'BARRED PUFFBIRD',
                   'BARROWS GOLDENEYE', 'BAY-BREASTED WARBLER', 'BEARDED BARBET', 'BEARDED BELLBIRD',
                   'BEARDED REEDLING', 'BELTED KINGFISHER', 'BIRD OF PARADISE', 'BLACK & YELLOW BROADBILL',
                   'BLACK BAZA', 'BLACK FRANCOLIN', 'BLACK SKIMMER', 'BLACK SWAN', 'BLACK TAIL CRAKE',
                   'BLACK THROATED BUSHTIT', 'BLACK THROATED WARBLER', 'BLACK VULTURE', 'BLACK-CAPPED CHICKADEE',
                   'BLACK-NECKED GREBE', 'BLACK-THROATED SPARROW', 'BLACKBURNIAM WARBLER', 'BLONDE CRESTED WOODPECKER',
                   'BLUE COAU', 'BLUE GROUSE', 'BLUE HERON', 'BLUE THROATED TOUCANET', 'BOBOLINK',
                   'BORNEAN BRISTLEHEAD', 'BORNEAN LEAFBIRD', 'BORNEAN PHEASANT', 'BRANDT CORMARANT', 'BROWN CREPPER',
                   'BROWN NOODY', 'BROWN THRASHER', 'BULWERS PHEASANT', 'BUSH TURKEY', 'CACTUS WREN',
                   'CALIFORNIA CONDOR', 'CALIFORNIA GULL', 'CALIFORNIA QUAIL', 'CANARY', 'CAPE GLOSSY STARLING',
                   'CAPE MAY WARBLER', 'CAPPED HERON', 'CAPUCHINBIRD', 'CARMINE BEE-EATER', 'CASPIAN TERN', 'CASSOWARY',
                   'CEDAR WAXWING', 'CERULEAN WARBLER', 'CHARA DE COLLAR', 'CHESTNET BELLIED EUPHONIA',
                   'CHIPPING SPARROW', 'CHUKAR PARTRIDGE', 'CINNAMON TEAL', 'CLARKS NUTCRACKER', 'COCK OF THE  ROCK',
                   'COCKATOO', 'COLLARED ARACARI', 'COMMON FIRECREST', 'COMMON GRACKLE', 'COMMON HOUSE MARTIN',
                   'COMMON LOON', 'COMMON POORWILL', 'COMMON STARLING', 'CRESTED AUKLET', 'CRESTED CARACARA',
                   'CRESTED NUTHATCH', 'CRIMSON SUNBIRD', 'CROW', 'CROWNED PIGEON', 'CUBAN TODY', 'CUBAN TROGON',
                   'CURL CRESTED ARACURI', 'D-ARNAUDS BARBET', 'DARK EYED JUNCO', 'DOUBLE BARRED FINCH',
                   'DOUBLE BRESTED CORMARANT', 'DOWNY WOODPECKER', 'EASTERN BLUEBIRD', 'EASTERN MEADOWLARK',
                   'EASTERN ROSELLA', 'EASTERN TOWEE', 'ELEGANT TROGON', 'ELLIOTS  PHEASANT', 'EMPEROR PENGUIN', 'EMU',
                   'ENGGANO MYNA', 'EURASIAN GOLDEN ORIOLE', 'EURASIAN MAGPIE', 'EVENING GROSBEAK', 'FAIRY BLUEBIRD',
                   'FIRE TAILLED MYZORNIS', 'FLAME TANAGER', 'FLAMINGO', 'FRIGATE', 'GAMBELS QUAIL',
                   'GANG GANG COCKATOO', 'GILA WOODPECKER', 'GILDED FLICKER', 'GLOSSY IBIS', 'GO AWAY BIRD',
                   'GOLD WING WARBLER', 'GOLDEN CHEEKED WARBLER', 'GOLDEN CHLOROPHONIA', 'GOLDEN EAGLE',
                   'GOLDEN PHEASANT', 'GOLDEN PIPIT', 'GOULDIAN FINCH', 'GRAY CATBIRD', 'GRAY KINGBIRD',
                   'GRAY PARTRIDGE', 'GREAT GRAY OWL', 'GREAT KISKADEE', 'GREAT POTOO', 'GREATOR SAGE GROUSE',
                   'GREEN BROADBILL', 'GREEN JAY', 'GREEN MAGPIE', 'GREY PLOVER', 'GROVED BILLED ANI', 'GUINEA TURACO',
                   'GUINEAFOWL', 'GYRFALCON', 'HARLEQUIN DUCK', 'HARPY EAGLE', 'HAWAIIAN GOOSE', 'HELMET VANGA',
                   'HIMALAYAN MONAL', 'HOATZIN', 'HOODED MERGANSER', 'HOOPOES', 'HORNBILL', 'HORNED GUAN',
                   'HORNED LARK', 'HORNED SUNGEM', 'HOUSE FINCH', 'HOUSE SPARROW', 'HYACINTH MACAW', 'IMPERIAL SHAQ',
                   'INCA TERN', 'INDIAN BUSTARD', 'INDIAN PITTA', 'INDIAN ROLLER', 'INDIGO BUNTING', 'IWI', 'JABIRU',
                   'JAVA SPARROW', 'KAGU', 'KAKAPO', 'KILLDEAR', 'KING VULTURE', 'KIWI', 'KOOKABURRA', 'LARK BUNTING',
                   'LAZULI BUNTING', 'LILAC ROLLER', 'LONG-EARED OWL', 'MAGPIE GOOSE', 'MALABAR HORNBILL',
                   'MALACHITE KINGFISHER', 'MALAGASY WHITE EYE', 'MALEO', 'MALLARD DUCK', 'MANDRIN DUCK',
                   'MANGROVE CUCKOO', 'MARABOU STORK', 'MASKED BOOBY', 'MASKED LAPWING', 'MIKADO  PHEASANT',
                   'MOURNING DOVE', 'MYNA', 'NICOBAR PIGEON', 'NOISY FRIARBIRD', 'NORTHERN CARDINAL',
                   'NORTHERN FLICKER', 'NORTHERN FULMAR', 'NORTHERN GANNET', 'NORTHERN GOSHAWK', 'NORTHERN JACANA',
                   'NORTHERN MOCKINGBIRD', 'NORTHERN PARULA', 'NORTHERN RED BISHOP', 'NORTHERN SHOVELER',
                   'OCELLATED TURKEY', 'OKINAWA RAIL', 'ORANGE BRESTED BUNTING', 'ORIENTAL BAY OWL', 'OSPREY',
                   'OSTRICH', 'OVENBIRD', 'OYSTER CATCHER', 'PAINTED BUNTING', 'PALILA', 'PARADISE TANAGER',
                   'PARAKETT  AKULET', 'PARUS MAJOR', 'PATAGONIAN SIERRA FINCH', 'PEACOCK', 'PELICAN',
                   'PEREGRINE FALCON', 'PHILIPPINE EAGLE', 'PINK ROBIN', 'POMARINE JAEGER', 'PUFFIN', 'PURPLE FINCH',
                   'PURPLE GALLINULE', 'PURPLE MARTIN', 'PURPLE SWAMPHEN', 'PYGMY KINGFISHER', 'QUETZAL',
                   'RAINBOW LORIKEET', 'RAZORBILL', 'RED BEARDED BEE EATER', 'RED BELLIED PITTA', 'RED BROWED FINCH',
                   'RED FACED CORMORANT', 'RED FACED WARBLER', 'RED FODY', 'RED HEADED DUCK', 'RED HEADED WOODPECKER',
                   'RED HONEY CREEPER', 'RED NAPED TROGON', 'RED TAILED HAWK', 'RED TAILED THRUSH',
                   'RED WINGED BLACKBIRD', 'RED WISKERED BULBUL', 'REGENT BOWERBIRD', 'RING-NECKED PHEASANT',
                   'ROADRUNNER', 'ROBIN', 'ROCK DOVE', 'ROSY FACED LOVEBIRD', 'ROUGH LEG BUZZARD', 'ROYAL FLYCATCHER',
                   'RUBY THROATED HUMMINGBIRD', 'RUDY KINGFISHER', 'RUFOUS KINGFISHER', 'RUFUOS MOTMOT',
                   'SAMATRAN THRUSH', 'SAND MARTIN', 'SANDHILL CRANE', 'SATYR TRAGOPAN', 'SCARLET CROWNED FRUIT DOVE',
                   'SCARLET IBIS', 'SCARLET MACAW', 'SCARLET TANAGER', 'SHOEBILL', 'SHORT BILLED DOWITCHER',
                   'SMITHS LONGSPUR', 'SNOWY EGRET', 'SNOWY OWL', 'SORA', 'SPANGLED COTINGA', 'SPLENDID WREN',
                   'SPOON BILED SANDPIPER', 'SPOONBILL', 'SPOTTED CATBIRD', 'SRI LANKA BLUE MAGPIE', 'STEAMER DUCK',
                   'STORK BILLED KINGFISHER', 'STRAWBERRY FINCH', 'STRIPED OWL', 'STRIPPED MANAKIN', 'STRIPPED SWALLOW',
                   'SUPERB STARLING', 'SWINHOES PHEASANT', 'TAIWAN MAGPIE', 'TAKAHE', 'TASMANIAN HEN', 'TEAL DUCK',
                   'TIT MOUSE', 'TOUCHAN', 'TOWNSENDS WARBLER', 'TREE SWALLOW', 'TROPICAL KINGBIRD', 'TRUMPTER SWAN',
                   'TURKEY VULTURE', 'TURQUOISE MOTMOT', 'UMBRELLA BIRD', 'VARIED THRUSH', 'VENEZUELIAN TROUPIAL',
                   'VERMILION FLYCATHER', 'VICTORIA CROWNED PIGEON', 'VIOLET GREEN SWALLOW', 'VULTURINE GUINEAFOWL',
                   'WALL CREAPER', 'WATTLED CURASSOW', 'WATTLED LAPWING', 'WHIMBREL', 'WHITE BROWED CRAKE',
                   'WHITE CHEEKED TURACO', 'WHITE NECKED RAVEN', 'WHITE TAILED TROPIC', 'WHITE THROATED BEE EATER',
                   'WILD TURKEY', 'WILSONS BIRD OF PARADISE', 'WOOD DUCK', 'YELLOW BELLIED FLOWERPECKER',
                   'YELLOW CACIQUE', 'YELLOW HEADED BLACKBIRD']
    # test_dir = "test"
    # for filename in os.listdir(test_dir):  # I used this section to get the list of class names exactly right
    #     class_names.append(str(filename))
    # print(class_names)                      # Then I hard-coded them above so that a user couldnt crash the
    #                                         # program by changing the contents of the test folder
    # temp_testing_function(Xception_model, class_names)
    # temp_testing_function(AlexNet_model, class_names)
    # temp_testing_function(ResNet_model, class_names)
    main_page_actions(class_names, Xception_model, AlexNet_model, ResNet_model)
