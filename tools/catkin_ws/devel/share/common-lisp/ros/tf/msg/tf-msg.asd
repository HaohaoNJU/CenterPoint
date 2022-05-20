
(cl:in-package :asdf)

(defsystem "tf-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "tfMessage" :depends-on ("_package_tfMessage"))
    (:file "_package_tfMessage" :depends-on ("_package"))
  ))