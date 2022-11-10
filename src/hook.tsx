import { Frame, useFrameProcessor } from 'react-native-vision-camera';
import { useState } from 'react';
import { runOnJS } from 'react-native-reanimated';
import { Face, FaceDetectionOptions, scanFaces } from '.';

export function useScanFaces(
  options?: FaceDetectionOptions
): [(frame: Frame) => void, Face[]] {
  const [faces, setFaces] = useState<Face[]>([]);
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    const detectedFaces = scanFaces(frame, options);
    runOnJS(setFaces)(detectedFaces);
  }, []);

  return [frameProcessor, faces];
}
