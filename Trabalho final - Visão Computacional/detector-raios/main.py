import cv2
import numpy as np
import time

def detecta_raio(img, frame_anterior, tempo_inicio=None, tempo_exibicao=1, threshold_movimento=100, tempo_ultimo_raio=None, threshold_area=5000, tempo_maximo=0.5):
    """
    Detecta a presença de raios, verificando movimentos repentinos e rápidos.
    Exibe a mensagem "Raio Detectado!" por um segundo após a detecção.
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

    raio_detectado = False

    if frame_anterior is not None:
        diff_img = cv2.absdiff(img_blur, frame_anterior)

        _, thresh = cv2.threshold(diff_img, threshold_movimento, 255, cv2.THRESH_BINARY)

        movimento_total = np.sum(thresh)

        if movimento_total > threshold_movimento:
            # Verifica se o movimento é muito grande e se a área de movimento é grande o suficiente
            area_movimento = cv2.countNonZero(thresh)

            if area_movimento > threshold_area:
                # Verifica se o movimento é muito rápido
                tempo_atual = time.time()
                if tempo_ultimo_raio is None or (tempo_atual - tempo_ultimo_raio) > tempo_maximo:  # Verifica se o movimento foi muito rápido (menos de 0.5 segundos)
                    raio_detectado = True
                    tempo_ultimo_raio = tempo_atual  
                    tempo_inicio = tempo_atual  

    if tempo_inicio is not None:
        tempo_decorrido = time.time() - tempo_inicio
        if tempo_decorrido < tempo_exibicao:
            cv2.putText(img, "Raio Detectado!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            tempo_inicio = None  

    frame_anterior = img_blur

    return img, raio_detectado, frame_anterior, tempo_inicio, tempo_ultimo_raio

def main():
    video_path = 'detector-raios/raio1.mp4'  
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    frame_anterior = None
    tempo_inicio = None 
    tempo_ultimo_raio = None  

    while True:
        check, img = video.read()
        if not check:
            break

        img_com_raio, raio_detectado, frame_anterior, tempo_inicio, tempo_ultimo_raio = detecta_raio(
            img, frame_anterior, tempo_inicio=tempo_inicio, tempo_ultimo_raio=tempo_ultimo_raio)

        cv2.imshow('Video', img_com_raio)

        if raio_detectado:
            print("Raio Detectado!")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
