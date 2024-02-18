import React, { useEffect, useState, useRef } from 'react';
import { StyleSheet, Text, View, Dimensions, Platform } from 'react-native';

import { Camera } from 'expo-camera';
import { CameraType } from 'expo-camera/build/Camera.types';

import * as tf from '@tensorflow/tfjs';
// import * as posedetection from '@tensorflow-models/pose-detection';
import * as ScreenOrientation from 'expo-screen-orientation';
import {
  bundleResourceIO,
  cameraWithTensors,
} from '@tensorflow/tfjs-react-native';
import Svg, { Circle } from 'react-native-svg';
import { ExpoWebGLRenderingContext } from 'expo-gl';

// tslint:disable-next-line: variable-name
const TensorCamera = cameraWithTensors(Camera);

const IS_ANDROID = Platform.OS === 'android';
const IS_IOS = Platform.OS === 'ios';

// Camera preview size.
//
// From experiments, to render camera feed without distortion, 16:9 ratio
// should be used fo iOS devices and 4:3 ratio should be used for android
// devices.
//
// This might not cover all cases.
const CAM_PREVIEW_WIDTH = Dimensions.get('window').width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// The score threshold for pose detection results.
const MIN_KEYPOINT_SCORE = 0.3;

// The size of the resized output from TensorCamera.
//
// For movenet, the size here doesn't matter too much because the model will
// preprocess the input (crop, resize, etc). For best result, use the size that
// doesn't distort the image.
const OUTPUT_TENSOR_WIDTH = 180;
const OUTPUT_TENSOR_HEIGHT = OUTPUT_TENSOR_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

const numClass = 30;

// Whether to auto-render TensorCamera preview.
const AUTO_RENDER = false;

// Whether to load model from app bundle (true) or through network (false).
const LOAD_MODEL_FROM_BUNDLE = false;

export default function App() {
  const cameraRef = useRef(null);
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState<tf.GraphModel>();
  const [preds, setPreds] = useState<tf.NamedTensorMap>();
  const [fps, setFps] = useState(0);
  const [orientation, setOrientation] =
    useState<ScreenOrientation.Orientation>();
  const [cameraType, setCameraType] = useState<CameraType>(
    CameraType.front
  );
  // Use `useRef` so that changing it won't trigger a re-render.
  //
  // - null: unset (initial value).
  // - 0: animation frame/loop has been canceled.
  // - >0: animation frame has been scheduled.
  const rafId = useRef<number | null>(null);

  useEffect(() => {
    async function prepare() {
      rafId.current = null;

      // Set initial orientation.
      const curOrientation = await ScreenOrientation.getOrientationAsync();
      setOrientation(curOrientation);

      // Listens to orientation change.
      ScreenOrientation.addOrientationChangeListener((event) => {
        setOrientation(event.orientationInfo.orientation);
      });

      // Camera permission.
      await Camera.requestCameraPermissionsAsync();

      // Wait for tfjs to initialize the backend.
      await tf.ready();

      //load custom tfjs model
      const modelJson = require('./yolov8/model.json');
      const modelWeights = [require('./yolov8/group1-shard1of11.bin'), require('./yolov8/group1-shard2of11.bin'), require('./yolov8/group1-shard3of11.bin'), require('./yolov8/group1-shard4of11.bin'), require('./yolov8/group1-shard5of11.bin'), require('./yolov8/group1-shard6of11.bin'), require('./yolov8/group1-shard7of11.bin'), require('./yolov8/group1-shard8of11.bin'), require('./yolov8/group1-shard9of11.bin'), require('./yolov8/group1-shard10of11.bin'), require('./yolov8/group1-shard11of11.bin')];

      const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
      setModel(model);

      // Ready!
      setTfReady(true);
    }

    prepare();
  }, []);

  useEffect(() => {
    // Called when the app is unmounted.
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);

  const preprocess = (source, modelWidth, modelHeight) => {
    let xRatio, yRatio; // ratios for boxes

    const input = tf.tidy(() => {
      const img = source;

      // padding image to square => [n, m] to [n, n], n > m
      const [h, w] = img.shape.slice(0, 2); // get source width and height
      const maxSize = Math.max(w, h); // get max size
      const imgPadded = img.pad([
        [0, maxSize - h], // padding y [bottom only]
        [0, maxSize - w], // padding x [right only]
        [0, 0],
      ]);

      xRatio = maxSize / w; // update xRatio
      yRatio = maxSize / h; // update yRatio

      return tf.image
        .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
        .div(255.0) // normalize
        .expandDims(0); // add batch
    });

    return [input, xRatio, yRatio];
  };

  const handleCameraStream = async (
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void,
    gl: ExpoWebGLRenderingContext
  ) => {
    const loop = async () => {
      // Get the tensor and run pose detection.
      const imageTensor = images.next().value as tf.Tensor3D;
      tf.engine().startScope();
      const [preprocImg, xRatio, yRatio] = preprocess(imageTensor, 800, 800);
      const startTs = Date.now();

      const res = model?.execute(preprocImg); // inference model
      const transRes = res?.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
      const boxes = tf.tidy(() => {
        const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
        const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
        const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
        const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
        return tf
          .concat(
            [
              y1,
              x1,
              tf.add(y1, h), //y2
              tf.add(x1, w), //x2
            ],
            2
          )
          .squeeze();
      }); // process boxes [y1, x1, y2, x2]

      const [scores, classes] = tf.tidy(() => {
        // class scores
        const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
        return [rawScores.max(1), rawScores.argMax(1)];
      }); // get max scores and classes index

      const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes

      const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
      const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
      const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

      tf.dispose([res, transRes, boxes, scores, classes, nms]);

      tf.engine().endScope();

      if (rafId.current === 0) {
        return;
      }

      // Render camera preview manually when autorender=false.
      if (!AUTO_RENDER) {
        updatePreview();
        gl.endFrameEXP();
      }

      const latency = Date.now() - startTs;
      setFps(Math.floor(1000 / latency));
      rafId.current = requestAnimationFrame(loop);
    };
    loop();
  };

  const renderPreds = () => {
    if (preds != null) {
      // const keypoints = preds[0].keypoints
      //   .filter((k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE)
      //   .map((k) => {
      //     // Flip horizontally on android or when using back camera on iOS.
      //     const flipX = IS_ANDROID || cameraType === CameraType.back;
      //     const x = flipX ? getOutputTensorWidth() - k.x : k.x;
      //     const y = k.y;
      //     const cx =
      //       (x / getOutputTensorWidth()) *
      //       (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
      //     const cy =
      //       (y / getOutputTensorHeight()) *
      //       (isPortrait() ? CAM_PREVIEW_HEIGHT : CAM_PREVIEW_WIDTH);
      //     return (
      //       <Circle
      //         key={`skeletonkp_${k.name}`}
      //         cx={cx}
      //         cy={cy}
      //         r='4'
      //         strokeWidth='2'
      //         fill='#00AA00'
      //         stroke='white'
      //       />
      //     );
      //   });

      return <Svg style={styles.svg}>{ }</Svg>;
    } else {
      return <View></View>;
    }
  };

  const renderFps = () => {
    return (
      <View style={styles.fpsContainer}>
        <Text>FPS: {fps}</Text>
      </View>
    );
  };

  const renderCameraTypeSwitcher = () => {
    return (
      <View
        style={styles.cameraTypeSwitcher}
        onTouchEnd={handleSwitchCameraType}
      >
        <Text>
          Switch to{' '}
          {cameraType === CameraType.front ? 'back' : 'front'} camera
        </Text>
      </View>
    );
  };

  const handleSwitchCameraType = () => {
    if (cameraType === CameraType.front) {
      setCameraType(CameraType.back);
    } else {
      setCameraType(CameraType.front);
    }
  };

  const isPortrait = () => {
    return (
      orientation === ScreenOrientation.Orientation.PORTRAIT_UP ||
      orientation === ScreenOrientation.Orientation.PORTRAIT_DOWN
    );
  };

  const getOutputTensorWidth = () => {
    // On iOS landscape mode, switch width and height of the output tensor to
    // get better result. Without this, the image stored in the output tensor
    // would be stretched too much.
    //
    // Same for getOutputTensorHeight below.
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_WIDTH
      : OUTPUT_TENSOR_HEIGHT;
  };

  const getOutputTensorHeight = () => {
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_HEIGHT
      : OUTPUT_TENSOR_WIDTH;
  };

  const getTextureRotationAngleInDegrees = () => {
    // On Android, the camera texture will rotate behind the scene as the phone
    // changes orientation, so we don't need to rotate it in TensorCamera.
    if (IS_ANDROID) {
      return 0;
    }

    // For iOS, the camera texture won't rotate automatically. Calculate the
    // rotation angles here which will be passed to TensorCamera to rotate it
    // internally.
    switch (orientation) {
      // Not supported on iOS as of 11/2021, but add it here just in case.
      case ScreenOrientation.Orientation.PORTRAIT_DOWN:
        return 180;
      case ScreenOrientation.Orientation.LANDSCAPE_LEFT:
        return cameraType === CameraType.front ? 270 : 90;
      case ScreenOrientation.Orientation.LANDSCAPE_RIGHT:
        return cameraType === CameraType.front ? 90 : 270;
      default:
        return 0;
    }
  };

  if (!tfReady) {
    return (
      <View style={styles.loadingMsg}>
        <Text>Loading...</Text>
      </View>
    );
  } else {
    let textureDims;
    if (Platform.OS === 'ios') {
      textureDims = {
        height: 1920,
        width: 1080,
      };
    } else {
      textureDims = {
        height: 1200,
        width: 1600,
      };
    }

    return (
      <View
        style={
          isPortrait() ? styles.containerPortrait : styles.containerLandscape
        }
      >
        <TensorCamera
          // Standard Camera props
          ref={cameraRef}
          style={styles.camera}
          autorender={AUTO_RENDER}
          type={cameraType}
          // tensor related props
          useCustomShadersToResize={true}
          cameraTextureHeight={textureDims.height}
          cameraTextureWidth={textureDims.width}
          resizeWidth={getOutputTensorWidth()}
          resizeHeight={getOutputTensorHeight()}
          resizeDepth={3}
          rotation={getTextureRotationAngleInDegrees()}
          onReady={handleCameraStream}
        />
        { }
        {renderFps()}
        {renderCameraTypeSwitcher()}
      </View>
    );
  }
}

const styles = StyleSheet.create({
  containerPortrait: {
    position: 'relative',
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    marginTop: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  containerLandscape: {
    position: 'relative',
    width: CAM_PREVIEW_HEIGHT,
    height: CAM_PREVIEW_WIDTH,
    marginLeft: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  loadingMsg: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    width: '100%',
    height: '100%',
    zIndex: 1,
  },
  svg: {
    width: '100%',
    height: '100%',
    position: 'absolute',
    zIndex: 30,
  },
  fpsContainer: {
    position: 'absolute',
    top: 10,
    left: 10,
    width: 80,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  cameraTypeSwitcher: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 180,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
});
