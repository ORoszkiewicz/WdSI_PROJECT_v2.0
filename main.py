import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import xml.etree.ElementTree as ET

def odczyt(traincondition):

    """
    Funkcja dokonująca odczytu zdjęć z folderu /images oraz opisów z plików .xml z folderu /annotations, lub z pliku input.txt
    @param traincondition: zmienna okreslajaca czy odczyt jest wykonywany dla potrzeb funkcji train czy predict
    @return: dane zawierające zdjęcie, koordynaty wycinku, nazwę pliku oraz jego label w przypadku przekazywania danych do funkcji train
    """

    dane = []
    if traincondition == 1 :
        sciezka_annotations = 'train/annotations'
        sciezka_images = 'train/images'
    else :
        sciezka_annotations = 'test/annotations'
        sciezka_images = 'test/images'

    if traincondition == 1: #dzialanie funkcji dla trenowania
        for filename in os.scandir(sciezka_annotations):
            if filename.is_file():
                tree = ET.parse(filename)
                root = tree.getroot()
                for country in root.findall('object'):  #country w tym miejscu pochodzi z przykladu ktorym sie posluzylem przy pisaniu kodu
                    filename = root[1].text             #spodobala mi sie ta nazwa i zdecydowalem sie jej nie zmieniac ;)
                    width = int(root[2][0].text)
                    height = int(root[2][1].text)
                    type = country.find('name').text
                    xmin = int(country.find('bndbox')[0].text)
                    ymin = int(country.find('bndbox')[1].text)
                    xmax = int(country.find('bndbox')[2].text)
                    ymax = int(country.find('bndbox')[3].text)

                    if type == 'speedlimit':
                        type = 1
                    elif type != 'speedlimit':
                        type = 0
                    #print(filename,width,heigh,type, xmin, ymin, xmax, ymax) #pomocnicze funkcje print do debugowania kodu
                    #print(root[4][5][0].text)       #

                    image = cv2.imread(os.path.join(sciezka_images, root.find('filename').text))
                    dane.append({'image': image, 'label': type, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'filename': filename})


    elif traincondition == 0 :  #dzialanie funkcji dla testowania
        wejscie = 'input.txt'
        with open(wejscie) as f:
            wejscie_lines = f.readlines()

        metoda_klasyfikacji = wejscie_lines[0].strip()

        #classify = input()
        if metoda_klasyfikacji == 'classify':

            i=0
            koniec=len(wejscie_lines)

            while i!=koniec:
                if i==0 or i==1:
                    i+=1
                    continue
                nazwa_pliku = wejscie_lines[i].strip()
                i+=1
                liczba_znakow=wejscie_lines[i].strip()
                i+=1
                image = cv2.imread(os.path.join(sciezka_images, nazwa_pliku))
                for j in range (0, int(liczba_znakow)):
                    coords = wejscie_lines[i].split(' ')
                    xmin = int(coords[0])
                    xmax = int(coords[1])
                    ymin = int(coords[2])
                    ymax = int(coords[3])
                    dane.append({'image': image, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'filename': nazwa_pliku})
                    i+=1

    return dane

def learn_bovw(dane):
    """
    Jest to jedynie lekko zmodyfikowana funkcja względem tej którą pisaliśmy na zajęciach podczas ostatnich laboratoriów.
    Learns BoVW dictionary and saves it as "voc.npy" file.
    @param data: List of dictionaries, one for every sample, with entries "image" and "label".
    @return: Nothing
    """
    bow = cv2.BOWKMeansTrainer(128)
    sift = cv2.SIFT_create()

    for sample in dane:
        gray = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2GRAY) #przetwarzanie na odcienie szarosci

        left = sample['xmin']
        top = sample['ymax']
        right = sample['xmax']
        bottom = sample['ymin']

        crop_image = gray[bottom:top, left:right] #zdjecia sa wycinane

        kp = sift.detect(crop_image, None)
        kp, dsc = sift.compute(crop_image, kp)

        if dsc is not None:
            bow.add(dsc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)

def extract_features(dane):
    """
    Extracts features for given data and saves it as "dsc" entry.
    @param data: List of dictionaries, one for every sample, with entries "image" and "label".
    @return: Data with added descriptors for each sample.
    """
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)

    for sample in dane:
        # compute descriptor and add it as "dsc" entry in sample

        zdjecie = sample['image']
        left = sample['xmin']
        top = sample['ymax']
        right = sample['xmax']
        bottom = sample['ymin']


        gray = cv2.cvtColor(zdjecie, cv2.COLOR_BGR2GRAY) #zdjecia sa przetwarzane do odcieni szarosci
        crop_image = gray[bottom:top, left:right] #ze zdjecia wycinany jest wycinek na ktorym znajduje sie analizowany obiekt
        # cv2.imshow(sample['filename'], gray) #fragment kodu sluzacy do debugowania
        # cv2.waitKey(0)

        kp = sift.detect(crop_image, None)
        dsc = bow.compute(crop_image, kp)
        sample['dsc'] = dsc

        # ------------------


    return dane


def train(dane):
    """
    Trains Random Forest classifier.
    @param data: List of dictionaries, one for every sample, with entries "image", "label", "desc".
    @return: Trained model.
    """
    # train random forest model and return it from function.

    klasyfikator = RandomForestClassifier(128) #wedle zrodel internetowych wartosc 128 podawana rfc jest najwyzszą proponowaną

    stack = np.empty((0, 128))
    typy = []

    for sample in dane:
        if sample['dsc'] is not None:
            stack = np.vstack((stack, sample['dsc'])) #probowalem tez nie uzywajac funkcji vstack (tylko innej) ale program nie chcial dzialac
            typy.append(sample['label'])

    klasyfikator.fit(stack[0:], typy)

    # ------------------

    return klasyfikator


def predict(klasyfikator, dane):
    """
    Predicts labels given a model and saves them as "predict" (int) entry for each sample. Prints out the results.
    @param rf: Trained model.
    @param data: List of dictionaries, one for every sample, with entries "image", "label" , "dsc".
    @return: Data with added predicted labels for each sample.
    """

    for sample in dane:
        if sample['dsc'] is not None:
            predict = klasyfikator.predict(sample['dsc'])
            sample['predict'] = predict
        else:
            sample['predict'] = None


        if sample['predict'] is not None:
            if sample['predict'] == 1:
                #print(sample['filename']) #print umozliwiajacy analize dla ktorego pliku zostala wykonana dana predykcja
                print("speedlimit")
            elif sample['predict'] == 0:
                #print(sample['filename']) #print umozliwiajacy analize dla ktorego pliku zostala wykonana dana predykcja
                print("other")
        else:
            print("other")
    # ------------------
    return dane


def main():

    #print('learning BoVW')
    data_train1 = odczyt(1)

    learn_bovw(data_train1)

    #print('extracting train features')
    data_train = extract_features(data_train1)

    #print('training')
    rf = train(data_train)

    #print('extracting test features')
    data_test = odczyt(0)
    data_test = extract_features(data_test)

    #print('testing on testing dataset')      #Wszelkie printy w mainie są zakomentowane, tak aby jedynymi wyjsciami z programu byly
    data_test = predict(rf, data_test)        #"speedlimit" i "other", ale printy te ułatwiają analizę programu oraz jego debugowanie

    return


if __name__ == '__main__':
    main()
