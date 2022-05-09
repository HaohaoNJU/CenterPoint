; Auto-generated. Do not edit!


(cl:in-package tf2_msgs-msg)


;//! \htmlinclude TF2Error.msg.html

(cl:defclass <TF2Error> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type cl:fixnum
    :initform 0)
   (error_string
    :reader error_string
    :initarg :error_string
    :type cl:string
    :initform ""))
)

(cl:defclass TF2Error (<TF2Error>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TF2Error>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TF2Error)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tf2_msgs-msg:<TF2Error> is deprecated: use tf2_msgs-msg:TF2Error instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <TF2Error>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-msg:error-val is deprecated.  Use tf2_msgs-msg:error instead.")
  (error m))

(cl:ensure-generic-function 'error_string-val :lambda-list '(m))
(cl:defmethod error_string-val ((m <TF2Error>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-msg:error_string-val is deprecated.  Use tf2_msgs-msg:error_string instead.")
  (error_string m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<TF2Error>)))
    "Constants for message type '<TF2Error>"
  '((:NO_ERROR . 0)
    (:LOOKUP_ERROR . 1)
    (:CONNECTIVITY_ERROR . 2)
    (:EXTRAPOLATION_ERROR . 3)
    (:INVALID_ARGUMENT_ERROR . 4)
    (:TIMEOUT_ERROR . 5)
    (:TRANSFORM_ERROR . 6))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'TF2Error)))
    "Constants for message type 'TF2Error"
  '((:NO_ERROR . 0)
    (:LOOKUP_ERROR . 1)
    (:CONNECTIVITY_ERROR . 2)
    (:EXTRAPOLATION_ERROR . 3)
    (:INVALID_ARGUMENT_ERROR . 4)
    (:TIMEOUT_ERROR . 5)
    (:TRANSFORM_ERROR . 6))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TF2Error>) ostream)
  "Serializes a message object of type '<TF2Error>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'error)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'error_string))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'error_string))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TF2Error>) istream)
  "Deserializes a message object of type '<TF2Error>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'error)) (cl:read-byte istream))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'error_string) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'error_string) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TF2Error>)))
  "Returns string type for a message object of type '<TF2Error>"
  "tf2_msgs/TF2Error")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TF2Error)))
  "Returns string type for a message object of type 'TF2Error"
  "tf2_msgs/TF2Error")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TF2Error>)))
  "Returns md5sum for a message object of type '<TF2Error>"
  "bc6848fd6fd750c92e38575618a4917d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TF2Error)))
  "Returns md5sum for a message object of type 'TF2Error"
  "bc6848fd6fd750c92e38575618a4917d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TF2Error>)))
  "Returns full string definition for message of type '<TF2Error>"
  (cl:format cl:nil "uint8 NO_ERROR = 0~%uint8 LOOKUP_ERROR = 1~%uint8 CONNECTIVITY_ERROR = 2~%uint8 EXTRAPOLATION_ERROR = 3~%uint8 INVALID_ARGUMENT_ERROR = 4~%uint8 TIMEOUT_ERROR = 5~%uint8 TRANSFORM_ERROR = 6~%~%uint8 error~%string error_string~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TF2Error)))
  "Returns full string definition for message of type 'TF2Error"
  (cl:format cl:nil "uint8 NO_ERROR = 0~%uint8 LOOKUP_ERROR = 1~%uint8 CONNECTIVITY_ERROR = 2~%uint8 EXTRAPOLATION_ERROR = 3~%uint8 INVALID_ARGUMENT_ERROR = 4~%uint8 TIMEOUT_ERROR = 5~%uint8 TRANSFORM_ERROR = 6~%~%uint8 error~%string error_string~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TF2Error>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'error_string))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TF2Error>))
  "Converts a ROS message object to a list"
  (cl:list 'TF2Error
    (cl:cons ':error (error msg))
    (cl:cons ':error_string (error_string msg))
))
