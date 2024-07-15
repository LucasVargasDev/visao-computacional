import cv2
import numpy as np

DELAY = int(1000 / 60) #Calculo para rodar em 60FPS
LARGURA, ALTURA = 800, 600 #Video é muito grande, ajustado para ficar em uma escala menor.
POSICAO_LINHA = 490 #Ficou fixo, mas se o tamanho do video fosse alterado era preciso fazer um calculo bom base no seu W e Y
HISTORICO_FRAMES = 10
AREA_MINIMA_CONTORNO = 3000  # Ajuste conforme necessário

def processa_frame(img):
    """
    Processamento semelhante ao utilizado no vagas
    """
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converte em cinza reduzindo a complexidade
    img_threshold = cv2.adaptiveThreshold(img_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16) #Deixa Objetos mais claros em fundos escuros

    return  img_threshold, img_cinza

def detecta_aviao(img_atual, img_anterior):
    diff = cv2.absdiff(img_anterior, img_atual)
    threshold_value, img_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    return cv2.dilate(img_thresh, None, iterations=2)

def encontra_contorno(img_dil):
    contornos, _ = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_filtrados = [contorno for contorno in contornos if cv2.contourArea(contorno) > AREA_MINIMA_CONTORNO]
    if contornos_filtrados:
        maior_contorno = max(contornos_filtrados, key=cv2.contourArea)
        return cv2.boundingRect(maior_contorno)

    return None 

def suaviza_contorno(historico_posicoes):
    if not historico_posicoes:
        return None
    media = np.mean(historico_posicoes, axis=0).astype(int)

    return tuple(media)

def main():
    video = cv2.VideoCapture('rastreio-pousos-avioes/pouso.mp4')
    if not video.isOpened():
        print(f"Erro ao abrir o vídeo: {video}")
        return

    ret, img_anterior = video.read()
    img_anterior = cv2.resize(img_anterior, (LARGURA, ALTURA))
    img_anterior_processado, _ = processa_frame(img_anterior)

    pousos = 0
    linha_cruzada = False
    aviao_passou = False
    historico_posicoes = []

    while True:
        ret, img_atual = video.read()
        if not ret:
            break

        img_atual = cv2.resize(img_atual, (LARGURA, ALTURA))
        img_atual_processado, img_cinza = processa_frame(img_atual)
        img_dilatada = detecta_aviao(img_atual_processado, img_anterior_processado)
        contorno = encontra_contorno(img_dilatada)

        if contorno and not aviao_passou:
            x, y, w, h = contorno
            historico_posicoes.append((x, y, w, h))

            if len(historico_posicoes) > HISTORICO_FRAMES:
                historico_posicoes.pop(0)

            contorno_suavizado = suaviza_contorno(historico_posicoes)

            if contorno_suavizado:
                #Desenha ao redor do avião
                cv2.rectangle(img_atual, (contorno_suavizado[0], contorno_suavizado[1]), 
                              (contorno_suavizado[0] + contorno_suavizado[2], contorno_suavizado[1] + contorno_suavizado[3]), 
                              (0, 255, 0), 2)

                #Verifica se o aviao cruzou a linha
                if contorno_suavizado[1] < POSICAO_LINHA < contorno_suavizado[1] + contorno_suavizado[3]:
                    if not linha_cruzada:
                        pousos += 1
                        linha_cruzada = True
                        aviao_passou = True
                else:
                    linha_cruzada = False

        cv2.line(img_atual, (0, POSICAO_LINHA), (LARGURA, POSICAO_LINHA), (0, 255, 0), 2)
        cv2.putText(img_atual, f'Pousos: {pousos}', (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)

        cv2.imshow('Video', img_atual)
        cv2.imshow('Imagem Processada', img_atual_processado)
        cv2.imshow('Imagem em Cinza', img_cinza)

        img_anterior_processado = img_atual_processado

        if cv2.waitKey(DELAY) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
