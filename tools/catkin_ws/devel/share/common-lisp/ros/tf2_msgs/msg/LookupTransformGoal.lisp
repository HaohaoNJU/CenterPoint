; Auto-generated. Do not edit!


(cl:in-package tf2_msgs-msg)


;//! \htmlinclude LookupTransformGoal.msg.html

(cl:defclass <LookupTransformGoal> (roslisp-msg-protocol:ros-message)
  ((target_frame
    :reader target_frame
    :initarg :target_frame
    :type cl:string
    :initform "")
   (source_frame
    :reader source_frame
    :initarg :source_frame
    :type cl:string
    :initform "")
   (source_time
    :reader source_time
    :initarg :source_time
    :type cl:real
    :initform 0)
   (timeout
    :reader timeout
    :initarg :timeout
    :type cl:real
    :initform 0)
   (target_time
    :reader target_time
    :initarg :target_time
    :type cl:real
    :initform 0)
   (fixed_frame
    :reader fixed_frame
    :initarg :fixed_frame
    :type cl:string
    :initform "")
   (advanced
    :reader advanced
    :initarg :advanced
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass LookupTransformGoal (<LookupTransformGoal>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <LookupTransformGoal>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'LookupTransformGoal)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tf2_msgs-msg:<LookupTransformGoal> is deprecated: use tf2_msgs-msg:LookupTransformGoal instead.")))

(cl:ensure-generic-function 'target_frame-val :lambda-list '(m))
(cl:defmethod target_frame-val ((m <LookupTransformGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-msg:target_frame-val is deprecated.  Use tf2_msgs-msg:target_frame instead.")
  (target_frame m))

(cl:ensure-generic-function 'source_frame-val :lambda-list '(m))
(cl:defmethod source_frame-val ((m <LookupTransformGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-msg:source_frame-val is deprecated.  Use tf2_msgs-msg:source_frame instead.")
  (source_frame m))

(cl:ensure-generic-function 'source_time-val :lambda-list '(m))
(cl:defmethod source_time-val ((m <LookupTransformGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-msg:source_time-val is deprecated.  Use tf2_msgs-msg:source_time instead.")
  (source_time m))

(cl:ensure-generic-function 'timeout-val :lambda-list '(m))
(cl:defmethod timeout-val ((m <LookupTransformGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-msg:timeout-val is deprecated.  Use tf2_msgs-msg:timeout instead.")
  (timeout m))

(cl:ensure-generic-function 'target_time-val :lambda-list '(m))
(cl:defmethod target_time-val ((m <LookupTransformGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-msg:target_time-val is deprecated.  Use tf2_msgs-msg:target_time instead.")
  (target_time m))

(cl:ensure-generic-function 'fixed_frame-val :lambda-list '(m))
(cl:defmethod fixed_frame-val ((m <LookupTransformGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-msg:fixed_frame-val is deprecated.  Use tf2_msgs-msg:fixed_frame instead.")
  (fixed_frame m))

(cl:ensure-generic-function 'advanced-val :lambda-list '(m))
(cl:defmethod advanced-val ((m <LookupTransformGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf2_msgs-msg:advanced-val is deprecated.  Use tf2_msgs-msg:advanced instead.")
  (advanced m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <LookupTransformGoal>) ostream)
  "Serializes a message object of type '<LookupTransformGoal>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'target_frame))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'target_frame))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'source_frame))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'source_frame))
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'source_time)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'source_time) (cl:floor (cl:slot-value msg 'source_time)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'timeout)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'timeout) (cl:floor (cl:slot-value msg 'timeout)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'target_time)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'target_time) (cl:floor (cl:slot-value msg 'target_time)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'fixed_frame))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'fixed_frame))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'advanced) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <LookupTransformGoal>) istream)
  "Deserializes a message object of type '<LookupTransformGoal>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'target_frame) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'target_frame) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'source_frame) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'source_frame) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'source_time) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'timeout) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'target_time) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'fixed_frame) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'fixed_frame) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:slot-value msg 'advanced) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<LookupTransformGoal>)))
  "Returns string type for a message object of type '<LookupTransformGoal>"
  "tf2_msgs/LookupTransformGoal")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'LookupTransformGoal)))
  "Returns string type for a message object of type 'LookupTransformGoal"
  "tf2_msgs/LookupTransformGoal")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<LookupTransformGoal>)))
  "Returns md5sum for a message object of type '<LookupTransformGoal>"
  "35e3720468131d675a18bb6f3e5f22f8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'LookupTransformGoal)))
  "Returns md5sum for a message object of type 'LookupTransformGoal"
  "35e3720468131d675a18bb6f3e5f22f8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<LookupTransformGoal>)))
  "Returns full string definition for message of type '<LookupTransformGoal>"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%#Simple API~%string target_frame~%string source_frame~%time source_time~%duration timeout~%~%#Advanced API~%time target_time~%string fixed_frame~%~%#Whether or not to use the advanced API~%bool advanced~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'LookupTransformGoal)))
  "Returns full string definition for message of type 'LookupTransformGoal"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%#Simple API~%string target_frame~%string source_frame~%time source_time~%duration timeout~%~%#Advanced API~%time target_time~%string fixed_frame~%~%#Whether or not to use the advanced API~%bool advanced~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <LookupTransformGoal>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'target_frame))
     4 (cl:length (cl:slot-value msg 'source_frame))
     8
     8
     8
     4 (cl:length (cl:slot-value msg 'fixed_frame))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <LookupTransformGoal>))
  "Converts a ROS message object to a list"
  (cl:list 'LookupTransformGoal
    (cl:cons ':target_frame (target_frame msg))
    (cl:cons ':source_frame (source_frame msg))
    (cl:cons ':source_time (source_time msg))
    (cl:cons ':timeout (timeout msg))
    (cl:cons ':target_time (target_time msg))
    (cl:cons ':fixed_frame (fixed_frame msg))
    (cl:cons ':advanced (advanced msg))
))
