import { Frame, useFrameProcessor } from '@better-network/react-native-vision-camera';
import { useState } from 'react';
import { runOnJS, SharedValue, useSharedValue } from 'react-native-reanimated';
import { Face, FaceDetectionOptions, scanFaces } from '.';

export function useScanFaces(
  options?: FaceDetectionOptions,
  width?: number
): [(frame: Frame) => void, Face[], SharedValue<{aspectRatio: number, width: number | undefined}>] {
  const sharedValue = useSharedValue<{aspectRatio: number, width: number | undefined}>({
    aspectRatio: 0,
    width: 0
  });
  const [faces, setFaces] = useState<Face[]>([]);
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet'
    const detectedFaces = scanFaces(frame, options);
    if(detectedFaces.length == 1){
      sharedValue.value = {
        aspectRatio: detectedFaces[0].bounds.aspectRatio, 
        width
      }
    }
    runOnJS(setFaces)(detectedFaces);
  }, []);

  return [frameProcessor, faces, sharedValue];
}
