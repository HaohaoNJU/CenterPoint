;; Auto-generated. Do not edit!


(when (boundp 'tf::FrameGraph)
  (if (not (find-package "TF"))
    (make-package "TF"))
  (shadow 'FrameGraph (find-package "TF")))
(unless (find-package "TF::FRAMEGRAPH")
  (make-package "TF::FRAMEGRAPH"))
(unless (find-package "TF::FRAMEGRAPHREQUEST")
  (make-package "TF::FRAMEGRAPHREQUEST"))
(unless (find-package "TF::FRAMEGRAPHRESPONSE")
  (make-package "TF::FRAMEGRAPHRESPONSE"))

(in-package "ROS")





(defclass tf::FrameGraphRequest
  :super ros::object
  :slots ())

(defmethod tf::FrameGraphRequest
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

(defclass tf::FrameGraphResponse
  :super ros::object
  :slots (_dot_graph ))

(defmethod tf::FrameGraphResponse
  (:init
   (&key
    ((:dot_graph __dot_graph) "")
    )
   (send-super :init)
   (setq _dot_graph (string __dot_graph))
   self)
  (:dot_graph
   (&optional __dot_graph)
   (if __dot_graph (setq _dot_graph __dot_graph)) _dot_graph)
  (:serialization-length
   ()
   (+
    ;; string _dot_graph
    4 (length _dot_graph)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _dot_graph
       (write-long (length _dot_graph) s) (princ _dot_graph s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _dot_graph
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _dot_graph (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tf::FrameGraph
  :super ros::object
  :slots ())

(setf (get tf::FrameGraph :md5sum-) "c4af9ac907e58e906eb0b6e3c58478c0")
(setf (get tf::FrameGraph :datatype-) "tf/FrameGraph")
(setf (get tf::FrameGraph :request) tf::FrameGraphRequest)
(setf (get tf::FrameGraph :response) tf::FrameGraphResponse)

(defmethod tf::FrameGraphRequest
  (:response () (instance tf::FrameGraphResponse :init)))

(setf (get tf::FrameGraphRequest :md5sum-) "c4af9ac907e58e906eb0b6e3c58478c0")
(setf (get tf::FrameGraphRequest :datatype-) "tf/FrameGraphRequest")
(setf (get tf::FrameGraphRequest :definition-)
      "---
string dot_graph

")

(setf (get tf::FrameGraphResponse :md5sum-) "c4af9ac907e58e906eb0b6e3c58478c0")
(setf (get tf::FrameGraphResponse :datatype-) "tf/FrameGraphResponse")
(setf (get tf::FrameGraphResponse :definition-)
      "---
string dot_graph

")



(provide :tf/FrameGraph "c4af9ac907e58e906eb0b6e3c58478c0")


