
(cl:in-package :asdf)

(defsystem "tf-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "FrameGraph" :depends-on ("_package_FrameGraph"))
    (:file "_package_FrameGraph" :depends-on ("_package"))
  ))