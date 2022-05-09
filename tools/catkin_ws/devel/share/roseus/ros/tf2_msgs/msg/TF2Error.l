;; Auto-generated. Do not edit!


(when (boundp 'tf2_msgs::TF2Error)
  (if (not (find-package "TF2_MSGS"))
    (make-package "TF2_MSGS"))
  (shadow 'TF2Error (find-package "TF2_MSGS")))
(unless (find-package "TF2_MSGS::TF2ERROR")
  (make-package "TF2_MSGS::TF2ERROR"))

(in-package "ROS")
;;//! \htmlinclude TF2Error.msg.html


(intern "*NO_ERROR*" (find-package "TF2_MSGS::TF2ERROR"))
(shadow '*NO_ERROR* (find-package "TF2_MSGS::TF2ERROR"))
(defconstant tf2_msgs::TF2Error::*NO_ERROR* 0)
(intern "*LOOKUP_ERROR*" (find-package "TF2_MSGS::TF2ERROR"))
(shadow '*LOOKUP_ERROR* (find-package "TF2_MSGS::TF2ERROR"))
(defconstant tf2_msgs::TF2Error::*LOOKUP_ERROR* 1)
(intern "*CONNECTIVITY_ERROR*" (find-package "TF2_MSGS::TF2ERROR"))
(shadow '*CONNECTIVITY_ERROR* (find-package "TF2_MSGS::TF2ERROR"))
(defconstant tf2_msgs::TF2Error::*CONNECTIVITY_ERROR* 2)
(intern "*EXTRAPOLATION_ERROR*" (find-package "TF2_MSGS::TF2ERROR"))
(shadow '*EXTRAPOLATION_ERROR* (find-package "TF2_MSGS::TF2ERROR"))
(defconstant tf2_msgs::TF2Error::*EXTRAPOLATION_ERROR* 3)
(intern "*INVALID_ARGUMENT_ERROR*" (find-package "TF2_MSGS::TF2ERROR"))
(shadow '*INVALID_ARGUMENT_ERROR* (find-package "TF2_MSGS::TF2ERROR"))
(defconstant tf2_msgs::TF2Error::*INVALID_ARGUMENT_ERROR* 4)
(intern "*TIMEOUT_ERROR*" (find-package "TF2_MSGS::TF2ERROR"))
(shadow '*TIMEOUT_ERROR* (find-package "TF2_MSGS::TF2ERROR"))
(defconstant tf2_msgs::TF2Error::*TIMEOUT_ERROR* 5)
(intern "*TRANSFORM_ERROR*" (find-package "TF2_MSGS::TF2ERROR"))
(shadow '*TRANSFORM_ERROR* (find-package "TF2_MSGS::TF2ERROR"))
(defconstant tf2_msgs::TF2Error::*TRANSFORM_ERROR* 6)
(defclass tf2_msgs::TF2Error
  :super ros::object
  :slots (_error _error_string ))

(defmethod tf2_msgs::TF2Error
  (:init
   (&key
    ((:error __error) 0)
    ((:error_string __error_string) "")
    )
   (send-super :init)
   (setq _error (round __error))
   (setq _error_string (string __error_string))
   self)
  (:error
   (&optional __error)
   (if __error (setq _error __error)) _error)
  (:error_string
   (&optional __error_string)
   (if __error_string (setq _error_string __error_string)) _error_string)
  (:serialization-length
   ()
   (+
    ;; uint8 _error
    1
    ;; string _error_string
    4 (length _error_string)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _error
       (write-byte _error s)
     ;; string _error_string
       (write-long (length _error_string) s) (princ _error_string s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _error
     (setq _error (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; string _error_string
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _error_string (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get tf2_msgs::TF2Error :md5sum-) "bc6848fd6fd750c92e38575618a4917d")
(setf (get tf2_msgs::TF2Error :datatype-) "tf2_msgs/TF2Error")
(setf (get tf2_msgs::TF2Error :definition-)
      "uint8 NO_ERROR = 0
uint8 LOOKUP_ERROR = 1
uint8 CONNECTIVITY_ERROR = 2
uint8 EXTRAPOLATION_ERROR = 3
uint8 INVALID_ARGUMENT_ERROR = 4
uint8 TIMEOUT_ERROR = 5
uint8 TRANSFORM_ERROR = 6

uint8 error
string error_string

")



(provide :tf2_msgs/TF2Error "bc6848fd6fd750c92e38575618a4917d")


