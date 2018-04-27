import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from PIL import Image
import time
np.set_printoptions(threshold=np.inf)


def glatting(u0, alpha=0.05):
    """
    Glatting eller "blurring" av bilde u0.
    :param u0: (str) path til bildet som skal glattes
    :param alpha: (int, optional) avgjør hvor fort glattingen utføres
    :return: intet
    """
    u0 = plt.imread(u0)
    plt.ion()
    data = plt.imshow(u0)
    plt.draw()
    t_end = time.time() + 7

    while time.time() < t_end:
        laplace = (u0[0:-2, 1:-1] +
                   u0[2:, 1:-1] +
                   u0[1:-1, 0:-2] +
                   u0[1:-1, 2:] -
                   4 * u0[1:-1, 1:-1])
        u0[1:-1, 1:-1] += alpha * laplace
        u0[:, 0] = u0[:, 1]
        u0[:, -1] = u0[:, -2]
        u0[0, :] = u0[1, :]
        u0[-1, :] = u0[-2, :]
        data.set_array(u0)
        plt.draw()
        plt.pause(1e-4)


def kontrast(src):
    """
    Forsterker kontrastene i bilde src
    :param src: (str) path til bilde som skal kontrast forsterkes

    :return: (Image object) kontrast forsterket bilde
    """
    th = 50
    u0 = lese_inn(src)
    hoyde, bredde, kanaler = u0.shape
    ui = np.zeros((hoyde, bredde, 3), dtype=np.uint8)

    for i in range(hoyde):                              # for-loop fordi np.where() ikke
        for j in range(bredde):                         # fungerer som jeg vil
            for k in range(3):                          # Lager mask basert på theta
                if u0[i, j, k] <= th:                   # Om mørkere enn 50, innenfor omega_i
                    ui[i, j, 0] = 255                   # om lysere utenfor omega_i
                    ui[i, j, 1] = 255
                    ui[i, k, 2] = 255

    lagrebildet(ui, 'kontrastmask')
    a, b, indx_til_koord, koord_til_indx, retning = initialisere(ui)
    return losning(u0, a, b, indx_til_koord, koord_til_indx, retning, u1='K')


def klone(src, mask, target):
    """
    Utfører sømløs kloning. Tar informasjon fra src bildet basert på masken og
    setter det inn i target. Informasjonen fra src blir endret slik at det passer
    inn i target bildet
    :param src: (str) path til forgrunnsbildet
    :param mask: (str) path til masken
    :param target: (str) path til bakgrunnsbildet
    :return: (Image object) ferdig klonet bilde
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
    Fyller inn informasjon innenfor omega_i basert på informasjon som grenser til
    omega_i i omega.
    :param src: (str) path til bilde
    :param mask: (str) path til mask
    :return: (Image object) inpainta bilde
    """
    u0, ui = lese_inn(src, mask, opr_type=1)
    if not like(u0, ui):
        raise ValueError('Masken og bildet maa ha samme hoyde og bredde')

    ui = np.where(ui > 170, 255, 0)
    a, b, indx_til_koord, koord_til_indx, retning = initialisere(ui)
    return losning(u0, a, b, indx_til_koord, koord_til_indx, retning)


def initialisere(ui):
    """
    Initialiserer variabler og indekserer omega_i
    Finner hvilke naboer i omega_i som også er i omega_i
    :param ui: (np.array) masken, 255 om innenfor omega_i 0 om utenfor
    :return: a (sparse array), b (array), indx_til_koord (list), koord_til_indx( array, retning (list)
    """

    hoyde = ui.shape[0]
    bredde = ui.shape[1]
    retning = []                    # Hvilken retning er omega og hvilken er omega_i
    indx_til_koord = []             # Indeks til koordinatene som er innenfor omega_i
    koord_til_indx = -1 * np.ones((hoyde, bredde))
                                    # indeksene der hvor omega_i befinner seg
    idx = 0
    for i in range(hoyde - 1):
        for j in range(bredde - 1):
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
    Setter opp poisson's ligning
    :param u0: (np.array) "bakgrunnsbildet"
    :param a:  (sparse array) laplace innenfor omega_i og i grenseområder
    :param b:  (np.array) nye verdiene
    :param indx_til_koord: (list) inneholder indeksene til koordinatene som er
            innenfor omega_i
    :param koord_til_indx: (np.array) inneholder koordinatene til indeksene som er
            innenfor omega_i
    :param retning: ( bool list) true innenfor omega_i false utenfor
    :param u1: (np.array, optional) "forgrunnsbildet"
    :return: a (sparse array) og b (array)
    """

    kont = False
    if u1 is None:
        u_x = u0
    elif u1 is 'K':
        kont = True
    else:
        u_x = u1

    for i in range(b.shape[0]):  # Initierer laplace matrise innenfor Omega_i og i grense området
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
            np.array(retning[j], dtype=int) + 1, 2)  # 0 -> innenfor u_i 1 -> utenfor u_i
        x, y = indx_til_koord[j]
        for m in range(3):
            if u1 is not None:
                b[j, m] = 4 * u0[x, y, m] - u0[x - 1, y, m] - u0[x + 1, y, m] - u0[x, y - 1, m] - \
                    u0[x, y + 1, m]
            if kont is False:
                b[j, m] += flag[0] * u_x[x - 1, y, m] + flag[1] * u_x[x + 1, y, m] + flag[2] * \
                    u_x[x, y - 1, m] + flag[3] * u_x[x, y + 1, m]
            if kont is True:
                b[j, m] -= 1.1 * (4 * u0[x, y, m] - u0[x - 1, y, m] - u0[x + 1, y, m] - u0[x, y - 1, m] -
                                  u0[x, y + 1, m])
    return a, b


def losning(u0, a, b, indx_til_koord, koord_til_indx, retning, u1=None):
    """
    Løser poissons ligning
    :param u0: (np.array) "bakgrunnsbildet"
    :param a:  (sparse array) laplace innenfor omega_i og på grensen
    :param b:  (np.array) verdiene innenfor omega_i
    :param indx_til_koord: (list) inneholder indeksene til koordinatene som er
            innenfor omega_i
    :param koord_til_indx: (np.array) inneholder koordinatene til indeksene som er
            innenfor omega_i
    :param retning: ( bool list) true innenfor omega_i false utenfor
    :param u1: (np.array eller str, optional) "forgrunnsbildet", 'K' om kontrastforskterkning
    :return: u (Image object) ferdig behandlet bilde
    """
    if u1 is None:
        u_x = u0
        a, b = eksplisitt(u0, a, b, indx_til_koord, koord_til_indx, retning)
    else:
        a, b = eksplisitt(u0, a, b, indx_til_koord, koord_til_indx, retning, u1=u1)
        u_x = u1

    c_r = linalg.cg(a, b[:, 0])[0]           # Løser ligning for hver kanal
    c_g = linalg.cg(a, b[:, 1])[0]
    c_b = linalg.cg(a, b[:, 2])[0]

    if u1 is 'K':                            # Om vi anvender kontrast forsterkning
        u = u0
    else:
        u = u_x

    for i in range(b.shape[0]):
        x, y = indx_til_koord[i]
        u[x, y, 0] = np.clip(c_r[i], 0, 255)  # min verdi 0
        u[x, y, 1] = np.clip(c_g[i], 0, 255)  # max verdi 255
        u[x, y, 2] = np.clip(c_b[i], 0, 255)  # legges inn i nytt bildet (u_i_i fylles inn)
    u = Image.fromarray(u)

    return u


def konverter(img):
    """
    Konverterer float64 array til int8 array
    :param img: (numpy array) array som skal konverteres fra float64 til int8
    :return: (numpy array) konvertert array
    """
    return 255 * img.astype(np.uint8)

def lagrebildet(img, filnavn):
    """
    Lagrer bildet til disk
    :param img: (array) array som skal lagres som bilde
    :param filnavn: (str) filnavn til bilde
    :return: intet
    """
    img = Image.fromarray(img)
    img.save(filnavn + '.jpg')


def like(img1, img2):
    """
    Sjekker om de to arrayene har lik størrelse
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
        print('')
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


def lagmaske(src):
    """
    Lager en binær maske av bildet. Område som skal maskes må
    markert i hvitt, resten kan være som det er.

    :param src: (str) path til bilde
    :return: (numpy array) ferdig maske
    """
    u = lese_inn(src)
    hoyde = u.shape[0]
    bredde = u.shape[1]
    for i in range(hoyde):
        for j in range(bredde):
            for k in range(3):
                if u[i, j, k] < 180:
                    u[i, j] = 0
                elif u[i, j, k] >= 180:
                    u[i, j] = 255

    return Image.fromarray(u)


if __name__ == "__main__":
    glatting('GreyCat.png')

    inp = inpaint('ferdig.jpg', 'maskds5.jpg')
    inp.save('inpaintgjov.jpg')

    kon = kontrast('ColorCat.png')
    kon.save('CatContrast.jpg')

    kl = klone('ds3.jpg', 'maskds4.jpg', 'gjov.jpg')
    kl.save('kloneferdig.jpg')

