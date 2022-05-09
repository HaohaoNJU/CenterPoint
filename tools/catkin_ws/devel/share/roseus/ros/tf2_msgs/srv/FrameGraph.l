;; Auto-generated. Do not edit!


(when (boundp 'tf2_msgs::FrameGraph)
  (if (not (find-package "TF2_MSGS"))
    (make-package "TF2_MSGS"))
  (shadow 'FrameGraph (find-package "TF2_MSGS")))
(unless (find-package "TF2_MSGS::FRAMEGRAPH")
  (make-package "TF2_MSGS::FRAMEGRAPH"))
(unless (find-package "TF2_MSGS::FRAMEGRAPHREQUEST")
  (make-package "TF2_MSGS::FRAMEGRAPHREQUEST"))
(unless (find-package "TF2_MSGS::FRAMEGRAPHRESPONSE")
  (make-package "TF2_MSGS::FRAMEGRAPHRESPONSE"))

(in-package "ROS")





(defclass tf2_msgs::FrameGraphRequest
  :super ros::object
  :slots ())

(defmethod tf2_msgs::FrameGraphRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass tf2_msgs::FrameGraphResponse
  :super ros::object
  :slots (_frame_yaml ))

(defmethod tf2_msgs::FrameGraphResponse
  (:init
   (&key
    ((:frame_yaml __frame_yaml) "")
    )
   (send-super :init)
   (setq _frame_yaml (string __frame_yaml))
   self)
  (:frame_yaml
   (&optional __frame_yaml)
   (if __frame_yaml (setq _frame_yaml __frame_yaml)) _frame_yaml)
  (:serialization-length
   ()
   (+
    ;; string _frame_yaml
    4 (length _frame_yaml)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _frame_yaml
       (write-long (length _frame_yaml) s) (princ _frame_yaml s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _frame_yaml
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _frame_yaml (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tf2_msgs::FrameGraph
  :super ros::object
  :slots ())

(setf (get tf2_msgs::FrameGraph :md5sum-) "437ea58e9463815a0d511c7326b686b0")
(setf (get tf2_msgs::FrameGraph :datatype-) "tf2_msgs/FrameGraph")
(setf (get tf2_msgs::FrameGraph :request) tf2_msgs::FrameGraphRequest)
(setf (get tf2_msgs::FrameGraph :response) tf2_msgs::FrameGraphResponse)

(defmethod tf2_msgs::FrameGraphRequest
  (:response () (instance tf2_msgs::FrameGraphResponse :init)))

(setf (get tf2_msgs::FrameGraphRequest :md5sum-) "437ea58e9463815a0d511c7326b686b0")
(setf (get tf2_msgs::FrameGraphRequest :datatype-) "tf2_msgs/FrameGraphRequest")
(setf (get tf2_msgs::FrameGraphRequest :definition-)
      "---
string frame_yaml

")

(setf (get tf2_msgs::FrameGraphResponse :md5sum-) "437ea58e9463815a0d511c7326b686b0")
(setf (get tf2_msgs::FrameGraphResponse :datatype-) "tf2_msgs/FrameGraphResponse")
(setf (get tf2_msgs::FrameGraphResponse :definition-)
      "---
string frame_yaml

")



(provide :tf2_msgs/FrameGraph "437ea58e9463815a0d511c7326b686b0")


