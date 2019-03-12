from time import sleep
import cv2
from Segmentation import Cap_Thread,Cap_Process
def test_thread():
    try:
        frames = Cap_Thread(8,6,0)
        while frames.is_alive():
            frame = frames.get(rgb=False)
            if frame is True :
                break
            cv2.imshow('frame',frame)          # The rest of code here(Any kind of processing is here)
            cv2.waitKey(30)
        frames.exit()
    except KeyboardInterrupt:
        print("Interrupt")
    finally:
        print('The programe is terminated ')
        cv2.destroyAllWindows()
        sleep(2)


def test_process():
    try:
        frames = Cap_Process(8,6,0)
        while frames.is_alive():
            frame = frames.get(rgb=False)
            if frame is True :
                break
            cv2.imshow('frame',frame)          # The rest of code here(Any kind of processing is here)
            cv2.waitKey(30)

    except KeyboardInterrupt:
        print("Interrupt")

    finally :
        frames.exit()
        print('The programe is terminated ')
        cv2.destroyAllWindows()
        sleep(2)

if __name__ == '__main__':
    test_process()
