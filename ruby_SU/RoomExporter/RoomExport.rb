# vim: set expandtab
# vim: set tabstop=2
##############################################################################
# This file is a part of PFFDTD.
# 
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
# 
# Copyright 2021 Brian Hamilton.
# 
# File name: RoomExport.rb
# 
# Description: Functions called by SU plugin (RoomExporter) to export geometry and materials
#   Exports Sketchup model's non-hidden faces and vertices to JSON
#    Uses Materials and not Layers (aka Tags in SU2020+)
#
# Notes:
# - model should be exploded prior to export (only exports top level entities)
# - internally Sketchup uses inches, this exports in metres.-
# - Materials should be one-sided (only one side has material), but can be two-sided if materials match
# - This plugin doesn't export/explode components or groups. That should be done beforehand.
# - Colors are exported, but textures are not (some average color somehow is exported for textures)
# 
##############################################################################

require 'sketchup.rb'
require 'fileutils.rb'
require 'csv'
require 'json'
#$DEBUG = true
$DEBUG = false
module RoomExporter

  INCHES2METRES = 0.0254
  MAX_SAFE_INTEGER = 9007199254740991

  module Export #main class
    def self.export_entities
      model = Sketchup.active_model
      ents = model.entities
      #initialise arrays
      mats_hash = {}
      counts_hash = {}
      counts_hash['n_groups'] = 0 #number of groups encountered
      counts_hash['n_comps'] = 0 #number of components encountered
      counts_hash['n_faces_hidden'] = 0 #number of hidden faces encountered
      counts_hash['n_faces_nvisible'] = 0 #number of faces not visible encountered
      counts_hash['n_faces_tofix'] = 0 #number of faces with some error
      counts_hash['n_faces_rigid'] = 0 #number of faces without materials
      Sketchup.set_status_text("Processing " + ents.length.to_s + " entities")
      for i in 0...ents.length 
        ent = ents[i]
        if ent.is_a?(Sketchup::Group) 
          Sketchup.set_status_text("Encountered Group")
          counts_hash['n_groups'] += 1 #ignore and notify later
        elsif ent.is_a?(Sketchup::ComponentInstance) 
          Sketchup.set_status_text("Encountered component: " + ent.definition.name)
          counts_hash['n_comps'] += 1 #ingore and notify later
        elsif ent.is_a?(Sketchup::Face) 
          #skip if hidden or visible
          if (ent.hidden?)
            counts_hash['n_faces_hidden'] += 1
            next
          end
          if !(ent.layer.visible?)
            counts_hash['n_faces_nvisible'] += 1
            next
          end

          has_fmat = false
          has_bmat = false
          bmat_name = ''
          fmat_name = ''

          #get material names
          if ent.back_material
            bmat_name = ent.back_material.display_name
            has_bmat = true
          end
          if ent.material
            fmat_name = ent.material.display_name
            has_fmat = true
          end

          #flag faces with two sides different materials
          if ((has_fmat && has_bmat) && (bmat_name != fmat_name))  #front and back face don't match
            if !(model.layers['_TOFIX'])
              layer = model.layers.add('_TOFIX')
            end
            puts ent.layer.name
            ent.layer = model.layers['_TOFIX'] #move to _TOFIX layer (= also assigns via layer.name)
            counts_hash['n_faces_tofix'] += 1
            next #skip to next entity
          end

          if (has_bmat) && (!has_fmat) #back-face (other side rigid)
            mat_name = bmat_name
            mat = ent.back_material
            mat_side = 1
          elsif (has_fmat) && (!has_bmat) #front-face (other side rigid)
            mat_name = fmat_name
            mat = ent.material
            mat_side = 2
          elsif (has_fmat) && (has_bmat) #two-sided
            mat_name = fmat_name
            mat = ent.material
            mat_side = 3
          else #no materials - rigid
            mat_name = '_RIGID' #has no material
            mat_side = 0
            counts_hash['n_faces_rigid'] += 1
          end

          if !mats_hash.has_key?(mat_name)
            mats_hash[mat_name] = {'tris' => [],
                                   'pts' => [], 
                                   'pt_cc' => 0, #point counter
                                   'color' => [0,0,0],
                                   'sides' => []
                                  }  #initialise empty array
          end
          #rigid has no material
          if !(mat_name=='_RIGID')
            mats_hash[mat_name]['color'] = mat.color.to_a[0..2] #save RGB
          end

          #get mesh of face
          mesh = ent.mesh(0)
          if mesh.points.empty? #skip if empty
            next
          end

          t_pts = mesh.points.to_a.map{|pt| pt.to_a.map{|d| d*INCHES2METRES}} #points for face mesh, convert to arrays
          #NB mesh.polygons are always triangles (sketchup triangulates for you)

          for j in 0...mesh.polygons.length

            vert_array = mesh.polygons[j]
            if vert_array.length != 3 #make sure is a triangle (and not an edge?)
              UI.messagebox('Non-triangle poly?' )
              next
            end

            tri = [vert_array[0].abs(), vert_array[1].abs(), vert_array[2].abs()]
            tri = tri.map{|t| t + mats_hash[mat_name]['pt_cc'] - 1} #1-indexing in Sketchup

            mats_hash[mat_name]['tris'] << tri
            mats_hash[mat_name]['sides'] << mat_side
          end
          mats_hash[mat_name]['pt_cc'] += t_pts.length
          mats_hash[mat_name]['pts'].concat(t_pts)

          if mats_hash[mat_name]['pt_cc']>MAX_SAFE_INTEGER #JSON uses JS 'Number' type, or double-precision (even for integers)
            UI.messagebox('Your model have have fewer than %d vertices to export.' % [MAX_SAFE_INTEGER])
            fail
          end
        end
      end

      #unique points for each material
      for mat_name in mats_hash.keys
        pts = mats_hash[mat_name]['pts']
        tris = mats_hash[mat_name]['tris']
        Sketchup.set_status_text("Sort-unique vertices...")
        unique_pts = Hash[pts.zip((0...pts.length))] #unique vertices
        ia = pts.map{|v| unique_pts[v]} #maps from pts to unique indices
        ib = Hash[unique_pts.values.zip((0...unique_pts.length))] #maps from unique indices in pts to indices of unique_pts
        tris = tris.map{|a| a.map{|i| ib[ia[i]]}} #remaps triangle indices
        pts = unique_pts.keys #saves smaller list (.keys() returns same order as .values())

        mats_hash[mat_name]['pts'] = pts
        mats_hash[mat_name]['tris'] = tris
        mats_hash[mat_name].delete('pt_cc') #cleanup
      end

      Sketchup.set_status_text("Processed " + ents.length.to_s + " entities")
      return mats_hash,counts_hash
    end

    def self.main #Export main
      model = Sketchup.active_model
      if (Sketchup.active_model.path.empty?)
        UI.messagebox 'Save your model in an appropriate directory first.'
        return
      end

      #go through model 
      Sketchup.set_status_text("Traversing model...")
      mats_hash, counts_hash = self.export_entities()
      Sketchup.set_status_text("Done traversing model...")

      ntris = mats_hash.map{|k,v| v['tris'].length}.inject(0, :+)
      npts = mats_hash.map{|k,v| v['pts'].length}.inject(0, :+)
      if $DEBUG
        puts 'ntris = %d' % [ntris]
        puts 'npts = %d' % [npts]
        puts 'nmaterials = %d' % [mats_hash.keys.length]
      end

      sources,warning_msg = SourceReceivers::read_CSV('source')
      if !warning_msg.empty?
        UI.messagebox(warning_msg)
      end
      receivers,warning_msg = SourceReceivers::read_CSV('receiver')
      if !warning_msg.empty?
        UI.messagebox(warning_msg)
      end

      json_hash = {'mats_hash': mats_hash, 
                  'sources': sources, 
                  'receivers': receivers, 
                  'export_datetime': Time.now}

      if $DEBUG
        #p mats_hash
        #p pts
        #p tris
        p json_hash
      end

      #write json from here
      Sketchup.set_status_text("Writing files...")

      #write in root dir
      write_dir = File.dirname(Sketchup.active_model.path)
      filename = 'model_export.json'
      file = File.new(write_dir + '\\' + filename, 'w')
      file.write(json_hash.to_json({object_nl: "\n"}))
      file.close
      Sketchup.set_status_text("Done.")

      #print status
      UI.messagebox('wrote %d points, %d tris, in %d materials to %s' % [npts, ntris, mats_hash.keys.length, write_dir])
      #print summary of materials (including any _RIGID)
      mats_msg = "Summary: \n"
      mkeys = mats_hash.keys
      for j in 0...mkeys.length
        mats_msg << mkeys[j] + ': ' + mats_hash[mkeys[j]]['tris'].length.to_s + " tris \n"
      end
      UI.messagebox(mats_msg)

      #print anything else worth noting
      if counts_hash['n_faces_hidden']>0 || counts_hash['n_faces_nvisible']>0 
        UI.messagebox('found %d hidden faces, %d faces not visible' % [counts_hash['n_faces_hidden'],counts_hash['n_faces_nvisible']])
      end
      if counts_hash['n_faces_tofix']>0 
        UI.messagebox('found %d faces with inconsistent material, moved to _TOFIX layer' % [counts_hash['n_faces_tofix']])
      end
      if counts_hash['n_faces_rigid']>0 
        UI.messagebox('found %d faces with no material, taken to be rigid' % [counts_hash['n_faces_rigid']])
      end
      if counts_hash['n_groups']>0 || counts_hash['n_comps']>0 
        UI.messagebox('found  %d groups, and %d components' % [counts_hash['n_groups'],counts_hash['n_comps']])
      end
      return
    end
  end

  module SourceReceivers
    def self.get_bounds_in_metres
      bmin = Geom::Point3d.new(Float::INFINITY,Float::INFINITY,Float::INFINITY)
      bmax = Geom::Point3d.new(-Float::INFINITY,-Float::INFINITY,-Float::INFINITY)
      model = Sketchup.active_model
      ents = model.entities
      for i in 0...ents.length 
        ent = ents[i]
        #only checks faces, not groups or components (only what's exported)
        if ent.is_a?(Sketchup::Face) 
          #skip if hidden or visible
          if (ent.hidden?)
            next
          end
          if !(ent.layer.visible?)
            next
          end
          bound = ent.bounds
          bmax.x = (bound.max.x>bmax.x) ? bound.max.x : bmax.x
          bmax.y = (bound.max.y>bmax.y) ? bound.max.y : bmax.y
          bmax.z = (bound.max.z>bmax.z) ? bound.max.z : bmax.z

          bmin.x = (bound.min.x<bmin.x) ? bound.min.x : bmin.x
          bmin.y = (bound.min.y<bmin.y) ? bound.min.y : bmin.y
          bmin.z = (bound.min.z<bmin.z) ? bound.min.z : bmin.z
        end
      end
      bmin_m = [bmin.x,bmin.y,bmin.z].map{|e| e*INCHES2METRES}
      bmax_m = [bmax.x,bmax.y,bmax.z].map{|e| e*INCHES2METRES}
      return bmin_m,bmax_m
    end

    def self.read_CSV(type) 
      warning_msg = ''
      srlist = []
      if type=='source'
        file = 'sources.csv'
        layer_name = '_SOURCES'
        prefix = 'S'
      elsif type=='receiver'
        file = 'receivers.csv'
        layer_name = '_RECEIVERS'
        prefix = 'R'
      end
      model = Sketchup.active_model
      root_dir = File.dirname(model.path)
      filename = root_dir + '\\' + file
      #puts filename
      if !(File.exists?(filename))
        error_msg = 'Error: \'' + file + '\' not found in SKP model directory'
        UI.messagebox(error_msg)
        return srlist,warning_msg
      end
      delimiters=[',',';',':','\t'] #will loop through to attempt different delimiters
      validCSV = false
      strip_converter = proc{|field| field.strip.upcase } #to strip headers of whitespace
      for delim in delimiters
        csv = CSV.read(filename, :headers => true, :col_sep => delim, :header_converters => strip_converter)
        headers = csv.headers
        #headers = csv.headers.map{|s| s.strip()}
        #puts headers
        if (headers - ['X','Y','Z','NAME']).empty?
          if csv.length>0
            validCSV = true
            break
          end
        end
      end
      if !validCSV
        error_msg = "Error: \'" + file + "\' is not a valid CSV file. \nFirst row needs column names: X,Y,Z,NAME \nDelimiter must be \',\' or \';\' or \':\' or Tab character.  Needs at least one entry."
        UI.messagebox(error_msg)
        return srlist,warning_msg
      end

      for i in 0...csv.length
        srlist << {}
        row = csv[i]
        srlist[i][:xyz] = [row['X'].to_f,row['Y'].to_f,row['Z'].to_f]
        srlist[i][:name] = ''
        if row['NAME'] != nil
          srlist[i][:name] = row['NAME']
        end
      end
      bmin,bmax = self.get_bounds_in_metres
      for i in 0...srlist.length
        xyz = srlist[i][:xyz]
        for j in 0...3
          if xyz[j]>bmax[j] || xyz[j]<bmin[j]
            warning_msg << 'Warning: ' + type +  ' at ' + xyz.map{|e| e.round(3)}.to_s + ' is outside of visible scene' + "\n"
            break
          end
        end
      end
      return srlist,warning_msg
    end

    def self.plot_positions(type) 
      if type=='source'
        layer_name = '_SOURCES'
        prefix = 'S'
      elsif type=='receiver'
        layer_name = '_RECEIVERS'
        prefix = 'R'
      end
      srlist,warning_msg = self.read_CSV(type)

      if !warning_msg.empty?
        UI.messagebox(warning_msg)
      end

      model = Sketchup.active_model
      if (model.layers[layer_name])
        model.layers.remove(layer_name,true)
      end
      layer = model.layers.add(layer_name)

      for i in 0...srlist.length
        xyz_in = srlist[i][:xyz].map{|e| e/INCHES2METRES}
        name = srlist[i][:name]
        pt = model.active_entities.add_cpoint(Geom::Point3d.new(xyz_in))
        pt.layer = layer
        if name == ''
          label = '―'+prefix+(i+1).to_s
        else
          label = '―'+prefix+(i+1).to_s + ':"' + name + '"'
        end
        text = model.active_entities.add_text(label,Geom::Point3d.new(xyz_in))
        text.layer = layer
      end
      return srlist
    end
    def self.plot
      sources = self.plot_positions('source')
      receivers = self.plot_positions('receiver')
      UI.messagebox('read %d '% [sources.length] +'sources and %d '% [receivers.length] +' receivers' )
    end
  end
end

version_required = 17 #this excepts Ruby version from 2017 and up
if (Sketchup.version.to_f < version_required)
  UI.messagebox("You must have Sketchup 20#{version_required} to run this extension. Visit sketchup.com to upgrade.")
  fail
end

#load once 
if (not file_loaded?('RoomExport.rb'))
  plugins_menu = UI.menu("Plugins")
  submenu = plugins_menu.add_submenu("Room Exporter")
  submenu.add_item("Export Geometry") { RoomExporter::Export.main() }
  submenu.add_item("Plot Sources and Receivers") { RoomExporter::SourceReceivers.plot() }
end
file_loaded('RoomExport.rb')
