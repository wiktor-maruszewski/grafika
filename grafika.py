import math
from enum import Enum
import numpy as np
from matplotlib.image import imread
from matplotlib.image import imsave
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import copy
from typing import Optional
import cv2


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # obraz 2d

class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu

    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, path: str = None) -> None:
        """
        inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        """
        if path != None:
            self.data = imread(path).astype('uint8')
            # self.data = cv2.imread(path)
            # im_rgb = self.data[:, :, [2, 1, 0]].astype('uint8')
            # self.data = im_rgb

            if self.data.ndim == 3:
                self.color_model = 0
            elif self.data.ndim == 2:
                self.color_model = 4

        pass


    def save_img(self, path: str) -> None:
        """
        metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        """

        if self.color_model == 0 or self.color_model == 4:
            imsave(path, self.data)
        else:
            self.to_rgb()
            imsave(path, self.data)

        pass

    def show_img(self) -> None:
        """
        metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        jezeli obraz jest w innym modelu barw, w celu samego wyswietlenia mozna uzyc
        from matplotlib.colors import hsv_to_rgb
        """

        if self.color_model == 1 and self.color_model == 2 and self.color_model == 3:
            self.to_rgb()

        if self.color_model == 4:
            imshow(self.data, cmap='gray')
        elif self.color_model == 0:
            imshow(self.data.astype('uint8'))


        plt.show()

        pass


    def get_layer(self, layer_id: int) -> 'BaseImage':
        """
        metoda zwracajaca warstwe o wskazanym indeksie
        """

        return self.data[:, :, layer_id]

        pass

    def to_hsv(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """

        if self.color_model != 0:
            self.to_rgb()

        M: np.ndarray = self.data.max(axis=2)
        m: np.ndarray = self.data.min(axis=2)
        V: np.ndarray = M / 255

        S: np.ndarray = np.zeros(M.shape)

        for i in range(0, S.shape[0]):
            for j in range(0, S.shape[1]):
                if M[i, j] > 0:
                    S[i, j] = 1 - (m[i, j] / M[i, j])
                else:
                    S[i, j] = 0

        H: np.ndarray = np.zeros(M.shape)

        R: np.ndarray = self.data[:, :, 0].astype("float32")
        G: np.ndarray = self.data[:, :, 1].astype("float32")
        B: np.ndarray = self.data[:, :, 2].astype("float32")

        for i in range(0, H.shape[0]):
            for j in range(0, H.shape[1]):

                temp: float = (R[i, j] - (1 / 2) * G[i, j] - (1 / 2) * B[i, j]) / (math.sqrt(
                    R[i, j] ** (2) + G[i, j] ** (2) + B[i, j] ** (2) - R[i, j] * G[i, j] - R[i, j] * B[i, j] - G[i, j] *
                    B[i, j]))

                if G[i, j] >= B[i, j]:
                    H[i, j] = math.degrees(math.acos(temp))
                else:
                    H[i, j] = 360 - math.degrees(math.acos(temp))

        self.data = np.stack((H, S, V), axis=2)
        self.color_model = 1
        return self


    def to_hsi(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """

        if self.color_model != 0:
            self.to_rgb()

        M: np.ndarray = self.data.max(axis=2)
        m: np.ndarray = self.data.min(axis=2)

        S: np.ndarray = np.zeros(M.shape)

        R: np.ndarray = self.data[:, :, 0].astype("float32")
        G: np.ndarray = self.data[:, :, 1].astype("float32")
        B: np.ndarray = self.data[:, :, 2].astype("float32")

        I: np.ndarray = np.zeros(R.shape).astype("float32")

        I = (R + G + B) / 3

        for i in range(0, S.shape[0]):
            for j in range(0, S.shape[1]):
                if M[i, j] > 0:
                    S[i, j] = 1 - (m[i, j] / I[i, j])
                else:
                    S[i, j] = 0

        H: np.ndarray = np.zeros(M.shape)


        for i in range(0, H.shape[0]):
            for j in range(0, H.shape[1]):

                temp: float = (R[i, j] - (1 / 2) * G[i, j] - (1 / 2) * B[i, j]) / (math.sqrt(
                    R[i, j] ** (2) + G[i, j] ** (2) + B[i, j] ** (2) - R[i, j] * G[i, j] - R[i, j] * B[i, j] - G[i, j] *
                    B[i, j]))

                if G[i, j] >= B[i, j]:
                    H[i, j] = math.degrees(math.acos(temp))
                else:
                    H[i, j] = 360 - math.degrees(math.acos(temp))


        self.data = np.stack((H, S, I), axis=2)
        self.color_model = 2
        return self

    def to_hsl(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """

        if self.color_model != 0:
            self.to_rgb()

        M: np.ndarray = self.data.max(axis=2).astype("float32")
        m: np.ndarray = self.data.min(axis=2).astype("float32")
        d: np.ndarray = np.zeros(M.shape).astype("float32")
        d = (M - m) / 255
        L: np.ndarray = np.zeros(M.shape).astype("float32")
        L = (0.5 * (M + m)) / 255

        S: np.ndarray = np.zeros(M.shape)

        for i in range(0, S.shape[0]):
            for j in range(0, S.shape[1]):
                if L[i, j] > 0:
                    S[i, j] = d[i, j] / (1 - abs(2 * L[i, j] - 1))
                else:
                    S[i, j] = 0

        H: np.ndarray = np.zeros(M.shape)

        R: np.ndarray = self.data[:, :, 0].astype("float32")
        G: np.ndarray = self.data[:, :, 1].astype("float32")
        B: np.ndarray = self.data[:, :, 2].astype("float32")

        for i in range(0, H.shape[0]):
            for j in range(0, H.shape[1]):

                temp: float = (R[i, j] - (1 / 2) * G[i, j] - (1 / 2) * B[i, j]) / (math.sqrt(
                    R[i, j] ** (2) + G[i, j] ** (2) + B[i, j] ** (2) - R[i, j] * G[i, j] - R[i, j] * B[i, j] - G[i, j] *
                    B[i, j]))

                if G[i, j] >= B[i, j]:
                    H[i, j] = math.degrees(math.acos(temp))
                else:
                    H[i, j] = 360 - math.degrees(math.acos(temp))

        self.data = np.stack((H, S, L), axis=2)
        self.color_model = 3
        return self

    def to_rgb(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """

        if self.color_model == 1:
            H: np.ndarray = self.data[:, :, 0].astype("float32")
            S: np.ndarray = self.data[:, :, 1].astype("float32")
            V: np.ndarray = self.data[:, :, 2].astype("float32")
            M: np.ndarray = np.zeros(H.shape).astype("float32")
            m: np.ndarray = np.zeros(H.shape).astype("float32")
            z: np.ndarray = np.zeros(H.shape).astype("float32")
            R: np.ndarray = np.zeros(H.shape).astype("float32")
            G: np.ndarray = np.zeros(H.shape).astype("float32")
            B: np.ndarray = np.zeros(H.shape).astype("float32")



            for i in range(0, H.shape[0]):
                for j in range(0, H.shape[1]):
                    M[i, j] = 255 * V[i, j]
                    m[i, j] = M[i, j] * (1 - S[i, j])
                    z[i, j] = (M[i, j] - m[i, j]) * (1 - abs((H[i, j] / 60.0) % 2.0 - 1))
                    if H[i, j] >= 0 and H[i, j] < 60:
                        R[i, j] = M[i, j]
                        G[i, j] = z[i, j] + m[i, j]
                        B[i, j] = m[i, j]
                    elif H[i, j] >= 60 and H[i, j] < 120:
                        R[i, j] = z[i, j] + m[i, j]
                        G[i, j] = M[i, j]
                        B[i, j] = m[i, j]
                    elif H[i, j] >= 120 and H[i, j] < 180:
                        R[i, j] = m[i, j]
                        G[i, j] = M[i, j]
                        B[i, j] = z[i, j] + m[i, j]
                    elif H[i, j] >= 180 and H[i, j] < 240:
                        R[i, j] = m[i, j]
                        G[i, j] = z[i, j] + m[i, j]
                        B[i, j] = M[i, j]
                    elif H[i, j] >= 240 and H[i, j] < 300:
                        R[i, j] = z[i, j] + m[i, j]
                        G[i, j] = m[i, j]
                        B[i, j] = M[i, j]
                    elif H[i, j] >= 300 and H[i, j] < 360:
                        R[i, j] = M[i, j]
                        G[i, j] = m[i, j]
                        B[i, j] = z[i, j] + m[i, j]


            self.data = np.stack((R.astype("uint8"), G.astype("uint8"), B.astype("uint8")), axis=2)
            self.color_model = 0
            return self

        elif self.color_model == 2:
            H: np.ndarray = self.data[:, :, 0].astype("float32")
            S: np.ndarray = self.data[:, :, 1].astype("float32")
            I: np.ndarray = self.data[:, :, 2].astype("float32")
            R: np.ndarray = np.zeros(H.shape).astype("float32")
            G: np.ndarray = np.zeros(H.shape).astype("float32")
            B: np.ndarray = np.zeros(H.shape).astype("float32")

            for i in range(0, H.shape[0]):
                for j in range(0, H.shape[1]):
                    if H[i, j] == 0:
                        R[i,j] = I[i,j] + 2 * I[i,j] * S[i,j]
                        G[i,j] = I[i,j] - I[i,j] * S[i,j]
                        B[i,j] = I[i,j] - I[i,j] * S[i,j]
                    elif H[i,j] > 0 and H[i,j] < 120:
                        R[i, j] = I[i, j] + I[i, j] * S[i, j] * math.cos(math.radians(H[i, j])) / math.cos(math.radians(60 - H[i, j]))
                        G[i, j] = I[i, j] + I[i, j] * S[i, j] * (1 - math.cos(math.radians(H[i, j])) / math.cos(math.radians(60 - H[i, j])))
                        B[i, j] = I[i, j] - I[i, j] * S[i, j]
                    elif H[i,j] == 120:
                        R[i, j] = I[i, j] - I[i, j] * S[i, j]
                        G[i, j] = I[i, j] + 2 * I[i, j] * S[i, j]
                        B[i, j] = I[i, j] - I[i, j] * S[i, j]
                    elif H[i,j] > 120 and H[i,j] < 240:
                        R[i, j] = I[i, j] - I[i, j] * S[i, j]
                        G[i, j] = I[i, j] + I[i, j] * S[i, j] * math.cos(math.radians(H[i, j] - 120)) / math.cos(math.radians(180 - H[i, j]))
                        B[i, j] = I[i, j] + I[i, j] * S[i, j] * (1 - math.cos(math.radians(H[i, j] - 120)) / math.cos(math.radians(180 - H[i, j])))
                    elif H[i,j] == 240:
                        R[i, j] = I[i, j] - I[i, j] * S[i, j]
                        G[i, j] = I[i, j] - I[i, j] * S[i, j]
                        B[i, j] = I[i, j] + 2 * I[i, j] * S[i, j]
                    elif H[i,j] > 240 and H[i,j] < 360:
                        R[i, j] = I[i, j] + I[i, j] * S[i, j] * (1 - math.cos(math.radians(H[i, j] - 240)) / math.cos(math.radians(300 - H[i, j])))
                        G[i, j] = I[i, j] - I[i, j] * S[i, j]
                        B[i, j] = I[i, j] + I[i, j] * S[i, j] * math.cos(math.radians(H[i, j] - 240)) / math.cos(math.radians(300 - H[i, j]))

            self.data = np.stack((R.astype("uint8"), G.astype("uint8"), B.astype("uint8")), axis=2)
            self.color_model = 0
            return self

        elif self.color_model == 3:
            H: np.ndarray = self.data[:, :, 0].astype("float32")
            S: np.ndarray = self.data[:, :, 1].astype("float32")
            L: np.ndarray = self.data[:, :, 2].astype("float32")
            R: np.ndarray = np.zeros(H.shape).astype("float32")
            G: np.ndarray = np.zeros(H.shape).astype("float32")
            B: np.ndarray = np.zeros(H.shape).astype("float32")
            d: np.ndarray = np.zeros(H.shape).astype("float32")
            m: np.ndarray = np.zeros(H.shape).astype("float32")
            x: np.ndarray = np.zeros(H.shape).astype("float32")

            for i in range(0, H.shape[0]):
                for j in range(0, H.shape[1]):
                    d[i,j] = S[i,j] * (1 - abs(2 * L[i,j] - 1))
                    m[i,j] = 255 * (L[i,j] - 0.5 * d[i,j])
                    x[i,j] = d[i,j] * (1 - abs(((H[i,j] / 60) % 2) - 1))

                    if H[i, j] >= 0 and H[i, j] < 60:
                        R[i, j] = 255 * d[i, j] + m[i, j]
                        G[i, j] = 255 * x[i, j] + m[i, j]
                        B[i, j] = m[i, j]
                    elif H[i, j] >= 60 and H[i, j] < 120:
                        R[i, j] = 255 * x[i, j] + m[i, j]
                        G[i, j] = 255 * d[i, j] + m[i, j]
                        B[i, j] = m[i, j]
                    elif H[i, j] >= 120 and H[i, j] < 180:
                        R[i, j] = m[i, j]
                        G[i, j] = 255 * d[i, j] + m[i, j]
                        B[i, j] = 255 * x[i, j] + m[i, j]
                    elif H[i, j] >= 180 and H[i, j] < 240:
                        R[i, j] = m[i, j]
                        G[i, j] = 255 * d[i, j] + m[i, j]
                        B[i, j] = 255 * x[i, j] + m[i, j]
                    elif H[i, j] >= 240 and H[i, j] < 300:
                        R[i, j] = 255 * x[i, j] + m[i, j]
                        G[i, j] = m[i, j]
                        B[i, j] = 255 * d[i, j] + m[i, j]
                    elif H[i, j] >= 300 and H[i, j] < 360:
                        R[i, j] = 255 * d[i, j] + m[i, j]
                        G[i, j] = m[i, j]
                        B[i, j] = 255 * x[i, j] + m[i, j]


            self.data = np.stack((R.astype("uint8"), G.astype("uint8"), B.astype("uint8")), axis=2)
            self.color_model = 0
            return self


class GrayScaleTransform(BaseImage):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        pass

    def to_gray(self) -> BaseImage:
        """
        metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        """

        temp: BaseImage = BaseImage()
        temp.color_model = self.color_model
        temp.data = np.zeros(self.data[:, :, 0].shape).astype("float32")

        R: np.ndarray = self.data[:, :, 0].astype("float32")
        G: np.ndarray = self.data[:, :, 1].astype("float32")
        B: np.ndarray = self.data[:, :, 2].astype("float32")

        for i in range(0, temp.data.shape[0]):
            for j in range(0, temp.data.shape[1]):
                temp.data[i, j] = ((R[i,j] + G[i, j] + B[i, j]) / 3.0).astype(int)

        temp.color_model = 4

        return temp

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        """
        metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        lub metoda 2 w przypadku przekazania argumentu w
        """
        temp: BaseImage = self.to_gray()
        L0: np.ndarray = temp.data.astype("float32")
        L1: np.ndarray = temp.data.astype("float32")
        L2: np.ndarray = temp.data.astype("float32")

        if alpha_beta != (None, None) and w is None:
            L0 = L0 * alpha_beta[0]
            L2 = L2 * alpha_beta[1]
        else:
            L0 = L0 + (2 * w)
            L1 = L1 + w

        L0[L0 > 255] = 255
        L1[L1 > 255] = 255

        temp.data = np.stack((L0.astype("uint8"), L1.astype("uint8"), L2.astype("uint8")), axis=2)
        temp.color_model = 0

        return temp

        pass


class Image(GrayScaleTransform):
    """
    klasa stanowiaca glowny interfejs biblioteki
    w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach
    realizujacych kolejne metody przetwarzania obrazow
    """


class Histogram:
    """
    klasa reprezentujaca histogram danego obrazu
    """
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray) -> None:
        if len(values.shape) == 3:
            self.values = np.ones((3,256))
            r = values[:, :, 0]
            g = values[:, :, 1]
            b = values[:, :, 2]
            r_hist = np.histogram(r, bins=range(0, 257))
            g_hist = np.histogram(g, bins=range(0, 257))
            b_hist = np.histogram(b, bins=range(0, 257))
            self.values[0, :] = r_hist[0]
            self.values[1, :] = g_hist[0]
            self.values[2, :] = b_hist[0]

        if len(values.shape) == 2:
            self.values = np.ones(256)
            g = values
            g_hist = np.histogram(g, bins=range(0, 257))
            self.values = g_hist[0]


    def plot(self) -> None:
        """
        metoda wyswietlajaca histogram na podstawie atrybutu values
        """
        if len(self.values.shape) == 2:
            plt.figure(figsize=[10, 5])
            plt.subplot(1, 3, 1)
            plt.plot(np.linspace(0, 255, 256), self.values[0, :], color='r')
            plt.subplot(1, 3, 2)
            plt.plot(np.linspace(0, 255, 256), self.values[1, :], color='g')
            plt.subplot(1, 3, 3)
            plt.plot(np.linspace(0, 255, 256), self.values[2, :], color='b')
            plt.show()

        if len(self.values.shape) == 1:
            plt.plot(np.linspace(0, 255, 256), self.values, color='b')
            plt.show()


    def to_cumulated(self) -> 'Histogram':
        """
        metoda zwracajaca histogram skumulowany na podstawie stanu wewnetrznego obiektu
        """

        if len(self.values.shape) == 2:
            his_temp = Histogram(np.ones((3, 256)))
            temp = np.ones((3, 256))
            temp[0, :] = np.cumsum(self.values[0, :])
            temp[1, :] = np.cumsum(self.values[1, :])
            temp[2, :] = np.cumsum(self.values[2, :])

            his_temp.values = temp
            return his_temp

            #x: np.ndarray = np.array([1,2,3,4,5])
            #print(np.cumsum(x))

        if len(self.values.shape) == 1:
            his_temp = Histogram(np.ones((256)))
            temp = np.ones((256))
            temp = np.cumsum(self.values)

            his_temp.values = temp
            return his_temp


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1

class ImageComparison(BaseImage):
    """
    Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    """

    def histogram(self) -> Histogram:
        """
        metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        """
        return Histogram(self.data)

    def compare_to(self, other: Image, method: ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """

        x: np.ndarray = (self.data[:, :, 0].astype("float32") + self.data[:, :, 1].astype("float32") +
                         self.data[:, :, 2].astype("float32")) / 3.0
        y: np.ndarray = other.to_gray().data

        x_hist = np.histogram(x.flatten(), bins=256, range=(0, 256))[0]
        y_hist = np.histogram(y.flatten(), bins=256, range=(0, 256))[0]

        mse: float = 0
        for i in range(0, 256):
            mse = mse + ((x_hist[i] - y_hist[i])**2)

        mse = mse / 256

        if method == 0:
            return mse
        else:
            return math.sqrt(mse)


class ImageAligning(BaseImage):
    """
    klasa odpowiadająca za wyrównywanie hostogramu
    """
    def __init__(self, path: str) -> None:
        """
        inicjalizator ...
        """
        super().__init__(path)
        pass

    @staticmethod
    def temp_tail_elimination_2d(ar: np.ndarray) -> np.ndarray:
        histogram = Histogram(ar)
        histt = histogram.to_cumulated()
        hist_data = histt.values

        print(hist_data)

        temp = np.copy(ar)

        all_value = ar.shape[0] * ar.shape[1]
        min_tail_value = 0.05 * all_value
        max_tail_value = 0.95 * all_value

        min_idx = np.min(np.where(hist_data > min_tail_value))
        max_idx = np.max(np.where(hist_data < max_tail_value))

        for i in range(0, temp.shape[0]):
            for j in range(0, temp.shape[1]):
                o = (temp[i, j] - min_idx) * (255 / (max_idx - min_idx))
                if o > 255:
                    o = 255

                if o < 0:
                    o = 0

                temp[i, j] = o

        print('Eliminacja ogonow min_idx:')
        print(min_idx)
        print('Eliminacja ogonow max_idx:')
        print(max_idx)

        return temp

    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        """

        temp: BaseImage = copy.deepcopy(self)

        if tail_elimination is False:
            if len(self.data.shape) == 2:
                for i in range(0, self.data.shape[0]):
                    for j in range(0, self.data.shape[1]):
                        temp.data[i, j] = (self.data[i, j] - self.data.min()) * (255/(self.data.max() - self.data.min()))

            if len(self.data.shape) == 3:
                r = np.copy(self.data[:, :, 0])
                g = np.copy(self.data[:, :, 1])
                b = np.copy(self.data[:, :, 2])

                r.setflags(write=1)
                g.setflags(write=1)
                b.setflags(write=1)

                for i in range(0, self.data.shape[0]):
                    for j in range(0, self.data.shape[1]):
                        r[i, j] = (r[i, j] - r.min()) * (255 / (r.max() - r.min()))

                for i in range(0, self.data.shape[0]):
                    for j in range(0, self.data.shape[1]):
                        g[i, j] = (g[i, j] - g.min()) * (255 / (g.max() - g.min()))

                for i in range(0, self.data.shape[0]):
                    for j in range(0, self.data.shape[1]):
                        b[i, j] = (b[i, j] - b.min()) * (255 / (b.max() - b.min()))

                temp.data = np.stack((r, g, b), axis=2)

        if tail_elimination is True:
            histogram = Histogram(self.data)
            hist = histogram.to_cumulated()
            hist_data = hist.values

            if len(self.data.shape) == 2:
                all_value = self.data.shape[0]*self.data.shape[1]
                min_tail_value = 0.05*all_value
                max_tail_value = 0.95*all_value

                min_idx = np.min(np.where(hist_data > min_tail_value))
                max_idx = np.max(np.where(hist_data < max_tail_value))

                for i in range(0, self.data.shape[0]):
                    for j in range(0, self.data.shape[1]):
                        temp.data[i, j] = (self.data[i, j] - min_idx) * (255/(max_idx - min_idx))


            if len(self.data.shape) == 3:
                r = self.temp_tail_elimination_2d(self.data[:, :, 0])
                g = self.temp_tail_elimination_2d(self.data[:, :, 1])
                b = self.temp_tail_elimination_2d(self.data[:, :, 2])

                temp.data = np.stack((r, g, b), axis=2)

        return temp


class ImageFiltration:
    def conv_2d(self, image: BaseImage, kernel: np.ndarray, prefix: Optional[float] = None) -> BaseImage:
        """
        kernel: filtr w postaci tablicy numpy
        prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        metoda zwroci obraz po procesie filtrowania
        """

        image.data = image.data.astype('uint8')
        # kernel = kernel.astype('uint8')
        temp = copy.deepcopy(image)

        print(kernel)


        for m in range(0, 3):
            central_idx_distance = math.floor(kernel.shape[0] / 2)
            for i in range(central_idx_distance, image.data.shape[0] - central_idx_distance):
                for j in range(central_idx_distance, image.data.shape[1] - central_idx_distance):
                    summ: int = 0
                    kernel_y = -1
                    for k in range(i - central_idx_distance, i + central_idx_distance + 1):
                        kernel_y = kernel_y + 1
                        kernel_x = -1
                        for l in range(j - central_idx_distance, j + central_idx_distance + 1):
                            kernel_x = kernel_x + 1
                            check = image.data[k, l, m]
                            check2 = kernel[kernel_y, kernel_x]
                            summ = summ + image.data[k, l, m] * kernel[kernel_y, kernel_x]

                    if summ > 255:
                        summ = 255

                    if summ < 0:
                        summ = 0

                    temp.data[i, j, m] = summ

        temp.data = temp.data.astype('uint8')
        return temp


class Thresholding(BaseImage):
    def threshold(self, value: int) -> BaseImage:
        """
        metoda dokonujaca operacji segmentacji za pomoca binaryzacji
        """

        result: BaseImage = copy.deepcopy(self)
        temp = np.ones(self.data[:, :, 0].shape)

        if len(self.data.shape) == 3:
            temp2 = np.ones(self.data[:, :, 0].shape)
            for i in range(0, self.data.shape[0]):
                for j in range(0, self.data.shape[1]):
                    temp2[i, j] = (self.data[i, j, 0].astype('float32') + self.data[i, j, 1].astype('float32') + self.data[i, j, 2].astype('float32')) / 3.0

            self.data = temp2.astype('uint8')
            self.color_model = 4

        for i in range(0, self.data.shape[0]):
            for j in range(0, self.data.shape[1]):
                if self.data[i, j] < value:
                    temp[i, j] = 0
                else:
                    temp[i, j] = 255

        result.data = temp
        print(temp)
        result.color_model = 4

        return result


img: BaseImage = BaseImage('lena.jpg')

img.show_img()

x = Thresholding('lena.jpg')
x.threshold(127).show_img()
