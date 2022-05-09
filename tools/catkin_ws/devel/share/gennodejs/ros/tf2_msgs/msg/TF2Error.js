// Auto-generated. Do not edit!

// (in-package tf2_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class TF2Error {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.error = null;
      this.error_string = null;
    }
    else {
      if (initObj.hasOwnProperty('error')) {
        this.error = initObj.error
      }
      else {
        this.error = 0;
      }
      if (initObj.hasOwnProperty('error_string')) {
        this.error_string = initObj.error_string
      }
      else {
        this.error_string = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TF2Error
    // Serialize message field [error]
    bufferOffset = _serializer.uint8(obj.error, buffer, bufferOffset);
    // Serialize message field [error_string]
    bufferOffset = _serializer.string(obj.error_string, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TF2Error
    let len;
    let data = new TF2Error(null);
    // Deserialize message field [error]
    data.error = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [error_string]
    data.error_string = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.error_string.length;
    return length + 5;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tf2_msgs/TF2Error';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'bc6848fd6fd750c92e38575618a4917d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 NO_ERROR = 0
    uint8 LOOKUP_ERROR = 1
    uint8 CONNECTIVITY_ERROR = 2
    uint8 EXTRAPOLATION_ERROR = 3
    uint8 INVALID_ARGUMENT_ERROR = 4
    uint8 TIMEOUT_ERROR = 5
    uint8 TRANSFORM_ERROR = 6
    
    uint8 error
    string error_string
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TF2Error(null);
    if (msg.error !== undefined) {
      resolved.error = msg.error;
    }
    else {
      resolved.error = 0
    }

    if (msg.error_string !== undefined) {
      resolved.error_string = msg.error_string;
    }
    else {
      resolved.error_string = ''
    }

    return resolved;
    }
};

// Constants for message
TF2Error.Constants = {
  NO_ERROR: 0,
  LOOKUP_ERROR: 1,
  CONNECTIVITY_ERROR: 2,
  EXTRAPOLATION_ERROR: 3,
  INVALID_ARGUMENT_ERROR: 4,
  TIMEOUT_ERROR: 5,
  TRANSFORM_ERROR: 6,
}

module.exports = TF2Error;
