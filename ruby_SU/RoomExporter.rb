# encoding: UTF-8

require 'sketchup.rb'
require 'extensions.rb'

module RoomExportExtension
  extension = SketchupExtension.new("Room Exporter", #name
                                    "RoomExporter/RoomExport") #import file
  extension.version = "1.0.0"
  extension.creator = "Brian Hamilton"
  extension.copyright = "© Brian Hamilton 2021"
  extension.description = "Export geometry with materials to JSON."
  Sketchup.register_extension(extension, true)
end

