; Auto-generated. Do not edit!


(cl:in-package tf2_msgs-srv)


;//! \htmlinclude FrameGraph-request.msg.html

(cl:defclass <FrameGraph-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass FrameGraph-request (<FrameGraph-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <FrameGraph-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'FrameGraph-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tf2_msgs-srv:<FrameGraph-request> is deprecated: use tf2_msgs-srv:FrameGraph-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <FrameGraph-request>) ostream)
  "Serializes a message object of type '<FrameGraph-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <FrameGraph-request>) istream)
  "Deserializes a message object of type '<FrameGraph-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<FrameGraph-request>)))
  "Returns string type for a service object of type '<FrameGraph-request>"
  "tf2_msgs/FrameGraphRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'FrameGraph-request)))
  "Returns string type for a service object of type 'FrameGraph-request"
  "tf2_msgs/FrameGraphRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<FrameGraph-request>)))
  "Returns md5sum for a message object of type '<FrameGraph-request>"
  "437ea58e9463815a0d511c7326b686b0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'FrameGraph-request)))
  "Returns md5sum for a message object of type 'FrameGraph-request"
  "437ea58e9463815a0d511c7326b686b0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<FrameGraph-request>)))
  "Returns full string definition for message of type '<FrameGraph-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'FrameGraph-request)))
  "Returns full string definition for message of type 'FrameGraph-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <FrameGraph-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <FrameGraph-request>))
  "Converts a ROS message object to a list"
  (cl:list 'FrameGraph-request
))
;//! \htmlinclude FrameGraph-response.msg.html

(cl:defclass <FrameGraph-response> (roslisp-msg-protocol:ros-message)
  ((frame_yaml
    :reader frame_yaml
    :initarg :frame_yaml
    :type cl:string
    :initform ""))
)

(cl:defclass FrameGraph-response (<FrameGraph-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <FrameGraph-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'FrameGraph-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tf2_msgs-srv:<FrameGraph-response> is deprecated: use tf2_msgs-srv:FrameGraph-response instead.")))

(cl:ensure-generic-function 'frame_yaml-val :lambda-list '(m))
(cl:defmethod frame_yaml-val ((m <FrameGraph-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-srv:frame_yaml-val is deprecated.  Use tf2_msgs-srv:frame_yaml instead.")
  (frame_yaml m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <FrameGraph-response>) ostream)
  "Serializes a message object of type '<FrameGraph-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'frame_yaml))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'frame_yaml))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <FrameGraph-response>) istream)
  "Deserializes a message object of type '<FrameGraph-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'frame_yaml) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'frame_yaml) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<FrameGraph-response>)))
  "Returns string type for a service object of type '<FrameGraph-response>"
  "tf2_msgs/FrameGraphResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'FrameGraph-response)))
  "Returns string type for a service object of type 'FrameGraph-response"
  "tf2_msgs/FrameGraphResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<FrameGraph-response>)))
  "Returns md5sum for a message object of type '<FrameGraph-response>"
  "437ea58e9463815a0d511c7326b686b0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'FrameGraph-response)))
  "Returns md5sum for a message object of type 'FrameGraph-response"
  "437ea58e9463815a0d511c7326b686b0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<FrameGraph-response>)))
  "Returns full string definition for message of type '<FrameGraph-response>"
  (cl:format cl:nil "string frame_yaml~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'FrameGraph-response)))
  "Returns full string definition for message of type 'FrameGraph-response"
  (cl:format cl:nil "string frame_yaml~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <FrameGraph-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'frame_yaml))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <FrameGraph-response>))
  "Converts a ROS message object to a list"
  (cl:list 'FrameGraph-response
    (cl:cons ':frame_yaml (frame_yaml msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'FrameGraph)))
  'FrameGraph-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'FrameGraph)))
  'FrameGraph-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'FrameGraph)))
  "Returns string type for a service object of type '<FrameGraph>"
  "tf2_msgs/FrameGraph")