import Vision
import MLKitFaceDetection
import MLKitVision
import CoreML

import UIKit
import AVFoundation

@objc(VisionCameraFaceDetector)
public class VisionCameraFaceDetector: NSObject, FrameProcessorPluginBase {
    private static var faceDetector: FaceDetector! = nil;
    
    private static func processContours(from face: Face) -> [String:[[String:CGFloat]]] {
      let faceContoursTypes = [
        FaceContourType.face,
        FaceContourType.leftEyebrowTop,
        FaceContourType.leftEyebrowBottom,
        FaceContourType.rightEyebrowTop,
        FaceContourType.rightEyebrowBottom,
        FaceContourType.leftEye,
        FaceContourType.rightEye,
        FaceContourType.upperLipTop,
        FaceContourType.upperLipBottom,
        FaceContourType.lowerLipTop,
        FaceContourType.lowerLipBottom,
        FaceContourType.noseBridge,
        FaceContourType.noseBottom,
        FaceContourType.leftCheek,
        FaceContourType.rightCheek,
      ]
      
      let faceContoursTypesStrings = [
        "FACE",
        "LEFT_EYEBROW_TOP",
        "LEFT_EYEBROW_BOTTOM",
        "RIGHT_EYEBROW_TOP",
        "RIGHT_EYEBROW_BOTTOM",
        "LEFT_EYE",
        "RIGHT_EYE",
        "UPPER_LIP_TOP",
        "UPPER_LIP_BOTTOM",
        "LOWER_LIP_TOP",
        "LOWER_LIP_BOTTOM",
        "NOSE_BRIDGE",
        "NOSE_BOTTOM",
        "LEFT_CHEEK",
        "RIGHT_CHEEK",
      ];
      
      var faceContoursTypesMap: [String:[[String:CGFloat]]] = [:]
      
      for i in 0..<faceContoursTypes.count {
        let contour = face.contour(ofType: faceContoursTypes[i]);
        
        var pointsArray: [[String:CGFloat]] = []
        
        if let points = contour?.points {
          for point in points {
            let currentPointsMap = [
                "x": point.x,
                "y": point.y,
            ]
            
            pointsArray.append(currentPointsMap)
          }
          
          faceContoursTypesMap[faceContoursTypesStrings[i]] = pointsArray
        }
      }
      
      return faceContoursTypesMap
    }
    
    private static func processLandMarks(from face: Face) -> [String:[String: CGFloat?]] {
      let faceLandmarkTypes = [
        FaceLandmarkType.leftCheek,
        FaceLandmarkType.rightCheek,
        FaceLandmarkType.leftEye,
        FaceLandmarkType.rightEye,
        FaceLandmarkType.leftEar,
        FaceLandmarkType.rightEar,
        FaceLandmarkType.noseBase,
        FaceLandmarkType.mouthLeft,
        FaceLandmarkType.mouthRight,
        FaceLandmarkType.mouthBottom,
      ]
      
      let faceLandmarksTypesStrings = [
        "LEFT_CHEEK",
        "RIGHT_CHEEK",
        "LEFT_EYE",
        "RIGHT_EYE",
        "LEFT_EAR",
        "RIGHT_EAR",
        "NOSE_BASE",
        "MOUTH_LEFT",
        "MOUTH_RIGHT",
        "MOUTH_BOTTOM"
      ];
      
        var faceLandMarksTypesMap: [String: [String: CGFloat?]] = [:]
      
      for i in 0..<faceLandmarkTypes.count {
        let landmark = face.landmark(ofType: faceLandmarkTypes[i]);
        let position = [
            "x": landmark?.position.x,
            "y": landmark?.position.y
        ]
        faceLandMarksTypesMap[faceLandmarksTypesStrings[i]] = position
      }
      
      return faceLandMarksTypesMap
    }
    
    private static func processBoundingBox(from face: Face) -> [String:Any] {
        let frameRect = face.frame
//      The implementation from this github repo seems to work better for the frameRect
//      Github link -> https://github.com/a7medev/react-native-ml-kit/blob/main/face-detection/ios/FaceDetection.m
        return [
          "x":frameRect.origin.x,
          "y": frameRect.origin.y,
          "width": frameRect.size.width,
          "height": frameRect.size.height,
          "boundingCenterX": frameRect.midX,
          "boundingCenterY": frameRect.midY,
          "aspectRatio": frameRect.size.width / photoWidth!
        ]
    }
    
    @objc
    public static func callback(_ frame: Frame!, withArgs args: [Any]!) -> Any! {
        let config = getConfig(withArgs: args)
        if faceDetector == nil {
            initFD(config: config)
        }
        let image = VisionImage(buffer: frame.buffer)
        let photoWidth = MLImage(sampleBuffer: frame.buffer)?.width
        image.orientation = .up
        
        var faceAttributes: [Any] = []
        
        do {
            let faces: [Face] =  try faceDetector.results(in: image)
            if (!faces.isEmpty){
                for face in faces {
                    var map: [String: Any] = [:]
                    
                    map["rollAngle"] = face.headEulerAngleZ  // Head is tilted sideways rotZ degrees
                    map["pitchAngle"] = face.headEulerAngleX  // Head is rotated to the uptoward rotX degrees
                    map["yawAngle"] = face.headEulerAngleY   // Head is rotated to the right rotY degrees
                    map["leftEyeOpenProbability"] = face.leftEyeOpenProbability
                    map["rightEyeOpenProbability"] = face.rightEyeOpenProbability
                    map["smilingProbability"] = face.smilingProbability
                    map["bounds"] = processBoundingBox(from: face, photoWidth: photoWidth)
                    map["contours"] = processContours(from: face)
                    map["landMarks"] = processLandMarks(from: face)
                    
                    faceAttributes.append(map)
                }
            }
        } catch _ {
            return nil
        }
        return faceAttributes
    }
    
    static func initFD(config: [String:Any]?) {
        let options = FaceDetectorOptions()
        
        if config?["landmarkMode"] as? String == "all" {
            options.landmarkMode = .all
        }
        
        if config?["contourMode"] as? String == "all" {
            options.contourMode = .all
        }
        
        if config?["performanceMode"] as? String == "accurate" {
            // doesn't work in fast mode!, why?
            options.performanceMode = .accurate
        }
        
        if config?["classificationMode"] as? String == "all" {
            options.classificationMode = .all
        }
        
        if config?["minFaceSize"] as? Double != nil {
            options.minFaceSize = CGFloat(config?["minFaceSize"] as? Double ?? 0.1)
        }
       
        faceDetector = FaceDetector.faceDetector(options: options)
    }
    
    static func getConfig(withArgs args: [Any]!) -> [String:Any]! {
           if args.count > 0 {
               let config = args[0] as? [String:Any]
               return config
           }
           return nil
    }
}
