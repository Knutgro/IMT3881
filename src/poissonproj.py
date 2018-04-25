import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from PIL import Image
np.set_printoptions(threshold=np.inf)


def kontrast(src):
    """

    :param src:
    :param alpha:
    :param th:
    :return:
    """
    th = 50
    u0 = lese_inn(src)
    hoyde, bredde, kanaler = u0.shape
    ui = np.ones((hoyde, bredde))
    for i in range(hoyde):
        for j in range(bredde):
            for k in range(3):
                if u0[i, j, k] <= th:
                    ui[i, j] = 255
                else:
                    ui[i, j] = 0
    a, b, indx_til_koord, koord_til_indx, retning = initialisere(ui)
    return losning(u0, a, b, indx_til_koord, koord_til_indx, retning, u1='K')

def klone(src, mask, target):
    """

    :param src:
    :param mask:
    :param target:
    :return:
    """
    u0, ui, u1 = lese_inn(src, mask, target, opr_type=2)
    if not like(u0, ui) and like(u0, u1):
        raise ValueError('Masken, forgrunnsbildet og bakgrunnsbildet'
                         'må ha samme hoyde og bredde')

    ui = np.where(ui > 122, 255, 0)
    a, b, indx_til_koord, koord_til_indx, retning = initialisere(ui)
    return losning(u0, a, b, indx_til_koord, koord_til_indx, retning, u1=u1)


def inpaint(src, mask):
    """

    :param src: (str) path til bildet
    :param mask: (str) path til mask
    :return: (3d array) inpainta bildet
    """
    u0, ui = lese_inn(src, mask, opr_type=1)
    if not like(u0, ui):
        raise ValueError('Masken og bildet maa ha samme hoyde og bredde')

    ui = np.where(ui > 170, 255, 0)
    a, b, indx_til_koord, koord_til_indx, retning = initialisere(ui)
    return losning(u0, a, b, indx_til_koord, koord_til_indx, retning)


def initialisere(ui):
    """

    :param ui:
    :return:
    """

    hoyde = ui.shape[0]
    bredde = ui.shape[1]
    retning = []
    indx_til_koord = []
    koord_til_indx = -1 * np.ones((hoyde, bredde))

    idx = 0
    for i in range(hoyde):
        for j in range(bredde):
            if ui[i, j, 0] == 255:
                indx_til_koord.append([i, j])
                retning.append([
                    i > 0 and ui[i - 1, j, 0] == 255,
                    i < hoyde - 1 and ui[i + 1, j, 0] == 255,
                    j > 0 and ui[i, j - 1, 0] == 255,
                    j < bredde - 1 and ui[i, j + 1, 0] == 255
                ])

                koord_til_indx[i][j] = idx
                idx += 1

    b = np.zeros((idx, 3))
    a = sparse.lil_matrix((idx, idx), dtype=int)
    return a, b, indx_til_koord, koord_til_indx, retning


def eksplisitt(u0, a, b, indx_til_koord, koord_til_indx, retning, u1=None):
    """

    :param u0: (np.array) "bakgrunnsbildet"
    :param a:  (sparse array) gradientene
    :param b:  (np.array) laplace
    :param indx_til_koord: (list)
    :param koord_til_indx: (np.array)
    :param retning: (list)
    :param u1: (np.array, optional) "forgrunnsbildet"
    :return: a og b
    """

    kont = False
    if u1 is None:
        u_x = u0
    elif u1 == 'K':
        kont = True

    else:
        u_x = u1

    for i in range(b.shape[0]):
        a[i, i] = 4
        x, y = indx_til_koord[i]
        if retning[i][0]:
            a[i, int(koord_til_indx[x - 1, y])] = - 1
        if retning[i][1]:
            a[i, int(koord_til_indx[x + 1, y])] = - 1
        if retning[i][2]:
            a[i, int(koord_til_indx[x, y - 1])] = - 1
        if retning[i][3]:
            a[i, int(koord_til_indx[x, y + 1])] = - 1

    for j in range(b.shape[0]):
        flag = np.mod(
            np.array(retning[j], dtype=int) + 1, 2)
        x, y = indx_til_koord[j]
        for m in range(3):
            if u1 is not None:
                b[j, m] = 4 * u0[x, y, m] - u0[x - 1, y, m] - u0[x + 1, y, m] - u0[x, y - 1, m] - \
                    u0[x, y + 1, m]
            if kont is False:
                b[j, m] += flag[0] * u_x[x - 1, y, m] + flag[1] * u_x[x + 1, y, m] + flag[2] * \
                    u_x[x, y - 1, m] + flag[3] * u_x[x, y + 1, m]
            if kont is True:
                b[j, m] *= 2.5
    return a, b


def losning(u0, a, b, indx_til_koord, koord_til_indx, retning, u1=None):
    """

    :param u0: (np.array) "bakgrunnsbildet"
    :param a:  (sparse array) gradientene
    :param b:  (np.array) laplace
    :param indx_til_koord: (list)
    :param koord_til_indx: (np.array)
    :param retning: (list)
    :param u1: (np.array, optional) "forgrunnsbildet"
    :return: u (np.array) inpainta bildet
    """
    if u1 is None:
        u_x = u0
    else:
        u_x = u1
    a, b = eksplisitt(u0, a, b, indx_til_koord, koord_til_indx, retning, u1=None)
    x_r = linalg.cg(a, b[:, 0])[0]
    x_g = linalg.cg(a, b[:, 1])[0]
    x_b = linalg.cg(a, b[:, 2])[0]
    u = u_x

    for i in range(b.shape[0]):
        x, y = indx_til_koord[i]
        u[x, y, 0] = np.clip(x_r[i], 0, 255)
        u[x, y, 1] = np.clip(x_g[i], 0, 255)
        u[x, y, 2] = np.clip(x_b[i], 0, 255)

    u = Image.fromarray(u)

    return u


def konverter(img):
    """
    Konverterer float64 array til int8 array
    :param img: (numpy array) array som skal konverteres fra float64 til int8
    :return: (numpy array) konvertert array
    """
    return 255 * img.astype(np.uint8)


def like(img1, img2):
    """
    Skekker om de to arrayene har lik størrelse
    :param img1: (numpy array) array nummer en som skal sammenlignes
    :param img2: (numpy array) array nummer to som skal sammenlignes
    :return: (bool) true om de er like, false om de ikke er like
    """
    return img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]



def lese_inn(src, mask=None, target=None, opr_type=0):
    """
    Leser inn bilder og maske, om de er av float64 [0, 1] blir de konvertert til uint8 [0, 255]
    :param src: (str) path til bildet som skal leses inn
    :param mask: (str, optional) path til masken som skal leses inn. Defaulter til None.
    :param target: (str, optional): path til bilde som skal leses inn. Brukes til kloning.
                    Defaulter til none
    :param opr_type: (int, optional) Bestemmer hva som skal leses inn.
                    0 til kontrastforsterkning og glatting ?. 1 for inpainting og 2 for kloning.
                    Defaulter til 0
    :return: numpy array - u0 om type 0
                            u0 og ui om type 1
                            u0, ui og u1 om type 2
    """
    u0 = np.array(Image.open(src))
    if u0.dtype == 'float64':
        u0 = konverter(u0)
    if opr_type > 0:
        ui = np.array(Image.open(mask))
        if ui.dtype == 'float64':
            ui = konverter(ui)
        if opr_type > 1:
            u1 = np.array(Image.open(target))
            if u1.dtype == 'float64':
                u1 = konverter(u1)
            return u0, ui, u1
        else:
            return u0, ui
    else:
        return u0


if __name__ == "__main__":
    print('hello')
    s = inpaint('gjov.jpg', 'gjovmask.jpg')
    plt.imshow(s)
    plt.show()
