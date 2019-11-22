import numpy as np
import train, preprocess, utils


model = train.build_model()


def predict(pic_path, model_path):
    model.load_weights(model_path)
    data = np.empty((5, 30, 30, 3), dtype="uint8")
    raw_img = preprocess.load_img(pic_path)
    sub_imgs = preprocess.gen_sub_img(raw_img)
    for sub_index, img in enumerate(sub_imgs):
        data[sub_index, :, :, :] = img / 255

    out = model.predict(data)
    result = np.array([np.argmax(i) for i in out])
    return ''.join([utils.CAT2CHR[i] for i in result])


if __name__ == '__main__':
    answer = predict('samples/01436.jpg', 'weights/40.hdf5')
    print(answer)
