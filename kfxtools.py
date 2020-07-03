'''
KFXTOOLS.PY is a library for reading, writing, and analyzing r3d files
from KFX.

Based on kfx.py written by Tarek Yousef.
Maintained by Geir Ove Myhr <Geir.Ove.Myhr@lr.org>

Wiki page: http://competence/SCPwiki/index.php/Kfxtools.py
'''

import sys
import gzip
from numpy import *
import numpy as np
from re import sub
from warnings import warn


big=1e10
small=-big

class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class R3d(object):

    def __init__(self, x, y, z,
                 names, fields,
                 t=0, r3dtype=None, gzipped=False,
                 **kwargs):
        """Setup an r3d-object with all variables."""

        # Computed values
        self.shape = (len(x),len(y),len(z))
        self.nvar = len(names)

        # Given values
        self.coordinates = [x,y,z]
        self.names = names
        self.fields = fields
        self.t = t
        self.gzipped = gzipped

        # Accept legacy igrid kw variable instead of r3dtype
        if r3dtype is None:
            if kwargs.has_key('igrid'):
                # Use the igrid variable for backward compatibility
                self.r3dtype = igrid
            else:
                # Use default value 1 for r3dtype
                self.r3dtype = 1
        else:
            # Ignore legacy igrid variable whether specified or not
            self.r3dtype = r3dtype

        # If r3dtype == 2, both centers (x,y,z) and staggered (xs,ys,zs)
        # should be passed as input (as both are included in the file).
        if self.r3dtype == 2:
            try:
                self.coordinates.append(kwargs['xs']) # index 3
                self.coordinates.append(kwargs['ys']) # index 4
                self.coordinates.append(kwargs['zs']) # index 5
            except KeyError as e:
                msg = ("Keyword arguments xs, ys, and zs required for " +
                       "when r3dtype is 2.\n" + str(e) )
                raise TypeError(msg)

    # Properties for use without caring about r3dtype
    # x,y,z will always return cell centers (if there are cells)
    # or point values (relevant when converted from bullet monitors)
    # Should give correct results for all supported types
    # - Type 1 with border cells (return raw coordinates)
    # - Type 1 with even spacing, such as converted from bullet monitors
    #       (return raw coordinates)
    # - Type 4 with border cells (convert coordinates from staggered)
    #
    # xs,ys,sz will always give staggered coordinates (i.e. gridline
    # on positive side of cell).

    @property
    def x(self):
        return self._center_coords(0)

    @property
    def y(self):
        return self._center_coords(1)

    @property
    def z(self):
        return self._center_coords(2)

    @property
    def xs(self):
        return self._staggered_coords(0)

    @property
    def ys(self):
        return self._staggered_coords(1)

    @property
    def zs(self):
        return self._staggered_coords(2)

    def _center_coords(self, axis):
        """Return write-protected view of center coordinates on the given axis."""
        rawcoord = self.coordinates[axis]
        if self.r3dtype in (1,2):
            retview = rawcoord.view()
        elif self.r3dtype == 4:
            retview =  staggered_to_center(rawcoord)
        retview.setflags(write=False)
        return retview

    def _staggered_coords(self, axis):
        """Return write-protected view of staggered coordinates on the given axis"""
        if self.r3dtype == 1:
            # Compute staggered grid, asssuming that the first and
            # last cells are border cells of zero width
            # (this is not always a valid assumption for type 1 files)
            rawcoord_center = self.coordinates[axis]
            retview = cellcenters_to_gridlines(rawcoord_center)[1:]
        elif self.r3dtype == 2:
            # In a type 2 r3d, the x, y, and z staggered coordinates are
            # at index 3, 4, and 5.
            rawcoord_staggered = self.coordinates[axis + 3]
            retview =  rawcoord_staggered.view()
        elif self.r3dtype == 4:
            # In a type 4 r3d, the coordinates are already staggered
            rawcoord_staggered = self.coordinates[axis]
            retview = rawcoord_staggered.view()
        retview.setflags(write=False)
        return retview

    def check_consistency(self):
        """Check that all dimensions match.

        This method raises an Error if it finds the r3d object to be
        inconsistent. If all the tests succeed, it does nothing.
        """

        if self.shape != (len(self.x),len(self.y),len(self.z)):
            raise Error(
                        'shape (%d,%d,%d) ' % self.shape +
                        'and length of x,y,z (%d,%d,%d) ' %
                            (len(self.x),len(self.y),len(self.z)) +
                        'are not equal'
                        )

        if len(self.names) != len(self.fields):
            raise Error(
                        'Length of names list (%d) is not equal ' +
                        'to lenth of fields list (%d)' %
                        (len(self.names), len(self.fields))
                       )

        for idx in range(len(self.fields)):
            field = self.fields[idx]
            if len(field.shape) != 3:
                raise Error(
                            'Field %d is not 3-dimensional (has dimension %d)' %
                            (idx,len(field.shape))
                            )
            if self.shape != field.shape:
                fieldname = self.names[idx]
                raise Error(
                            (
                             'Dimensions of field %d: %s (%d,%d,%d) does ' +
                             'not match dimension of coordinates (%d,%d,%d)'
                             ) %
                               ((idx,fieldname) + field.shape + self.shape)
                           )

    def point_value(self, x, y, z, fieldnum):
        """Return the value of a field in a point, with trilinear interpolation.

        This essentially duplicates the functionality of fieldutil -pointval,
        but fieldutil unfortunately has a fixed-point six-decimal output,
        making it useless for small values.

        Return None if the given point is outside the coordinate box of the r3d
        file.

        """

        # Make sure coordinates are floats (needed for interpolation)
        x = float(x)
        y = float(y)
        z = float(z)

        if not ((self.x[0] < x < self.x[-1]) and
                (self.y[0] < y < self.y[-1]) and
                (self.z[0] < z < self.z[-1])):
            # Point is outside grid
            return None
        xidx = np.searchsorted(self.x, x, side='right')
        yidx = np.searchsorted(self.y, y, side='right')
        zidx = np.searchsorted(self.z, z, side='right')
        # 2x2x2 cube with values on the corners in which we do the interpolation
        V = self.fields[fieldnum][xidx-1:xidx+1, yidx-1:yidx+1, zidx-1:zidx+1]

        # Trilinear interpolation
        px = (x - self.x[xidx-1]) / (self.x[xidx] - self.x[xidx-1])
        py = (y - self.y[yidx-1]) / (self.y[yidx] - self.y[yidx-1])
        pz = (z - self.z[zidx-1]) / (self.z[zidx] - self.z[zidx-1])
        value = ( (1-px) * (1-py) * (1-pz) * V[0,0,0] +
                  (1-px) * (1-py) *   pz   * V[0,0,1] +
                  (1-px) *   py   * (1-pz) * V[0,1,0] +
                  (1-px) *   py   *   pz   * V[0,1,1] +
                    px   * (1-py) * (1-pz) * V[1,0,0] +
                    px   * (1-py) *   pz   * V[1,0,1] +
                    px   *   py   * (1-pz) * V[1,1,0] +
                    px   *   py   *   pz   * V[1,1,1] )
        return value

    def new_grid(self, x, y, z, **kwargs):
        """Return a new R3d instance with new grid.

        x, y, z : Coordinates of the new grid. 1-dimensional numpy arrays.

        Key word arguments:
        coords :    'centers' (default) if x, y, z represent new cell centers.
                    'gridlines' if x, y, z represent new grid lines. For
                    consistency with KFX and other methods in kfxtools.py,
                    the first two and the last two coordinates in each array
                    should be the same, representing zero-width cells on the
                    border. The output of compute_gridlines() is on this form.
                    'baregridlines' same as gridlines, but without the first
                    and last value duplicated. Makes it convenient to specify
                    gridline coordinates with numpy.linspace() or
                    numpy.arange().
        method :    'max' (default) new cell gets maximum value of all old
                    cells that overlap  with new cell
                    'min' new cell gets minimum value of all old cells that
                    overlap with new cell
                    'point' new cell value is the interpolated point value in
                    the cell center of the new cell (Not yet implemented).
        outsideval: Value for new cells that fall outside old grid. Default is
                    None, which raises an error if new cells are not inside
                    old domain.
        field_list: List of field indices that are converted. Indexing starts
                    at 0. Default is all fields.

        """

        # Parse kwargs
        try:
            coords = kwargs['coords']
        except KeyError:
            coords = 'centers'
        try:
            method = kwargs['method']
        except KeyError:
            method = 'max'
        try:
            outsideval = kwargs['outsideval']
        except KeyError:
            outsideval = None
        try:
            field_list = kwargs['field_list']
        except KeyError:
            field_list = range(self.nvar)

        # Accept now depreciated spelling of 'centers':
        if coords == 'centres':
            coords = 'centers'

        # Find cell centers and gridlines
        if coords == 'centers':
            center_x = x
            center_y = y
            center_z = z
            gridlines_x = cellcenters_to_gridlines(x)
            gridlines_y = cellcenters_to_gridlines(y)
            gridlines_z = cellcenters_to_gridlines(z)
        elif coords == 'gridlines':
            center_x = gridlines_to_cellcenters(x)
            center_y = gridlines_to_cellcenters(y)
            center_z = gridlines_to_cellcenters(z)
            gridlines_x = x
            gridlines_y = y
            gridlines_z = z
        elif coords == 'baregridlines':
            x = np.concatenate((x[:1],x,x[-1:]))
            y = np.concatenate((y[:1],y,y[-1:]))
            z = np.concatenate((z[:1],z,z[-1:]))
            center_x = gridlines_to_cellcenters(x)
            center_y = gridlines_to_cellcenters(y)
            center_z = gridlines_to_cellcenters(z)
            gridlines_x = x
            gridlines_y = y
            gridlines_z = z
        else:
            errstr = ('coords argument must be \'centers\', \'gridlines\'' +
                     ' or \'baregridlines\', not \'{0}\'')
            raise Error(errstr.format(coords))

        # Create the new R3d instance
        newr3d = R3d(center_x, center_y, center_z, [], [], t=self.t)

        # --- Prepare grid-dependent information ---

        if method == 'max' or method == 'min':

            # Below, i,j,k-like variables are used for indexing fields in
            # the old r3d instance (self). In the new r3d instance
            # (newr3d) indices l,m,n are used.

            # Find out which old coordinates each new coordinate overlaps
            # with. I.e. for a list of (1-dim) cells find lower and upper
            # index from the old grid.
            old_lines_x,old_lines_y,old_lines_z = self.compute_gridlines()
            imin_array = np.searchsorted(old_lines_x, gridlines_x[:-1], side='right') - 1
            imax_array = np.searchsorted(old_lines_x, gridlines_x[1:], side='left')
            jmin_array = np.searchsorted(old_lines_y, gridlines_y[:-1], side='right') - 1
            jmax_array = np.searchsorted(old_lines_y, gridlines_y[1:], side='left')
            kmin_array = np.searchsorted(old_lines_z, gridlines_z[:-1], side='right') - 1
            kmax_array = np.searchsorted(old_lines_z, gridlines_z[1:], side='left')

            # In case of degeneracy (a coordinate in gridlinex_* is equal
            # to two coordinates in old_lines_*), an empty list of cells
            # will be created for a given cell above.
            # Adjust so that any cell in the new grid gets its value from
            # the extremum over at least one cell in the old grid.
            imax_array = np.maximum(imax_array, imin_array + 1)
            jmax_array = np.maximum(jmax_array, jmin_array + 1)
            kmax_array = np.maximum(kmax_array, kmin_array + 1)

            # Error if there are new gridlines outside old domain
            # and no outsideval is set
            if outsideval is None:
                if imin_array[0] == -1:
                    raise Error('Lower x border outside old domain and no outsideval given')
                elif jmin_array[0] == -1:
                    raise Error('Lower y border outside old domain and no outsideval given')
                elif kmin_array[0] == -1:
                    raise Error('Lower z border outside old domain and no outsideval given')
                elif imax_array[-1] == self.shape[0] + 1:
                    raise Error('Upper x border outside old domain and no outsideval given')
                elif jmax_array[-1] == self.shape[1] + 1:
                    raise Error('Upper y border outside old domain and no outsideval given')
                elif kmax_array[-1] == self.shape[2] + 1:
                    raise Error('Upper z border outside old domain and no outsideval given')

            # Find range of coordinates such that the new cells have some
            # overlap with the old cells.
            # This will give 0 and newr3d.shape[*] if outsideval is not set
            # and we have not yet raised an error.
            lmin = np.searchsorted(imin_array, 0, side='left')
            lmax = np.searchsorted(imax_array, self.shape[0], side='right')
            mmin = np.searchsorted(jmin_array, 0, side='left')
            mmax = np.searchsorted(jmax_array, self.shape[1], side='right')
            nmin = np.searchsorted(kmin_array, 0, side='left')
            nmax = np.searchsorted(kmax_array, self.shape[2], side='right')

        elif method == 'point':
            raise Error('Method \'{0}\' not yet implemented'.format(method))
        else:
            raise Error('Method \'{0}\' not implemented'.format(method))

        # --- Compute the new fields ---

        # Regenerate each field with new grid
        for fieldidx in field_list:
            name = self.names[fieldidx]
            oldfield = self.fields[fieldidx]

            if outsideval is None:
                # Initialize empty array
                newfield = np.empty(newr3d.shape)
            else:
                # Initialize array to outsideval
                newfield = np.ones(newr3d.shape) * float(outsideval)

            if method == 'max' or method == 'min':

                # Use min_or_max() as name for min() or max()
                if method == 'max':
                    min_or_max = np.max
                else:
                    min_or_max = np.min

                # Loop over all new cells that are inside the old domain
                # This should be faster than looping over old cells,
                # provided that there are fewer new cells than old
                # cells within the new domain.
                for l in range(lmin,lmax):
                    imin = imin_array[l]
                    imax = imax_array[l]
                    for m in range(mmin,mmax):
                        jmin = jmin_array[m]
                        jmax = jmax_array[m]
                        for n in range(nmin,nmax):
                            kmin = kmin_array[n]
                            kmax = kmax_array[n]
                            # Finally - each new cell is the min/max of all old overlapping cells
                            newfield[l,m,n] = min_or_max(oldfield[imin:imax,jmin:jmax,kmin:kmax])

            elif method == 'point':
                raise Error('Method \'{0}\' not yet implemented'.format(method))
            else:
                raise Error('Method \'{0}\' not implemented'.format(method))

            # Add the new field to the new r3d object
            newr3d.add_field(newfield, name)

        # Return newly created r3d instance
        return newr3d

    def add_field(self,field,name):
        """Add another field to the r3d instance"""
        if field.shape != self.shape:
            raise Error(
                        'Dimension of added field (%d,%d,%d) does ' %
                            field.shape +
                        'not match pre-set values (%d,%d,%d)' %
                         self.shape
                       )
        self.names.append(name)
        self.fields.append(field)
        self.nvar = self.nvar + 1

    def write(self, filename, gzipped=None):
        """ Write R3d instance to r3d-file """

        from struct import pack

        # If gzipped option not specified, use that of the R3d object
        if gzipped is None:
            gzipped = self.gzipped

        # Open gzipped file or normal file
        if gzipped:
            rawfile = open(filename,'wb')
            # The headers of the gzipped r3d files produced by this is
            # a bit different from those produced by KFX:
            # +---------+------+--------------+
            # | Field   |  KFX | This module  |
            # +---------+------+--------------+
            # | MTIME   | 0    | current time |
            # | XFL     | 0    | 2            |
            # | OS      | 3    | 255          |
            # +-------------------------------+
            # See http://www.gzip.org/zlib/rfc-gzip.html for field definitions
            f = gzip.GzipFile(fileobj=rawfile, filename='')
        else:
            f = open(filename,'wb')

        # Write single variables
        f.write(pack("3i",self.shape[0],self.shape[1],self.shape[2]))
        f.write(pack("i",self.nvar))
        f.write(pack("i",self.r3dtype))
        f.write(pack("f",self.t))

        # Write the field names
        for i in range(self.nvar):
            f.write(pack("32s",self.names[i]))

        # Write coordinates - dependent on r3dtype
        if self.r3dtype in (1,4):
            f.write(pack(str(self.shape[0])+'f',*self.coordinates[0]))
            f.write(pack(str(self.shape[1])+'f',*self.coordinates[1]))
            f.write(pack(str(self.shape[2])+'f',*self.coordinates[2]))
        elif self.r3dtype == 2:
            f.write(pack(str(self.shape[0])+'f',*self.coordinates[0]))
            f.write(pack(str(self.shape[1])+'f',*self.coordinates[1]))
            f.write(pack(str(self.shape[2])+'f',*self.coordinates[2]))
            f.write(pack(str(self.shape[0])+'f',*self.coordinates[3]))
            f.write(pack(str(self.shape[1])+'f',*self.coordinates[4]))
            f.write(pack(str(self.shape[2])+'f',*self.coordinates[5]))
        else:
            msg = "r3dtype {0}".format(self.r3dtype)
            raise NotImplementedError(msg)


        # Write field arrays
        for i in range(self.nvar):
            formatstring = str(self.shape[0]*self.shape[1]*self.shape[2]) + 'f'
            packstring = pack(formatstring,*ravel(self.fields[i],order='F'))
            f.write(packstring)

        # Close gzipped file or normal file
        if gzipped:
            f.close()
            rawfile.close()
        else:
            f.close()

    def regularize_names(self):
        """Attempt to normalize the field names in the r3d file.

        This is rarely used, and have not seen much love.
        """
        for i in range(self.nvar):
            self.names[i]= self.names[i].replace(' ','_').split('(')[0]
            self.names[i] = sub('_*?$','',self.names[i])   #remove underscores after name

            # KFX uses a policy of random naming of variables ...
            # ... try to regularize
            self.names[i] = sub('vph','Volume_Porosity',self.names[i])
            self.names[i] = sub('VPH','Volume_Porosity',self.names[i])

            # Trim fuel names..   Fuel names are sometimes called Vol pr of ... etc
            # sometimes not.
            self.names[i] = sub('Vol_pr._of_','',self.names[i])

            # Some variable names include minus signs
            self.names[i] = sub('-','_',self.names[i])

    def get_field_by_name(self,fieldname):
        """Return the field with the given field name.

        Warning: The field names in KFX are not always consistent and
        sometimes contain extra spaces at the end.
        """
        for idx in range(len(self.names)):
            if fieldname == self.names[idx]:
                return self.fields[idx]
        else:
            raise Error("Field '%s' not found" % fieldname)

    def radar(self,fieldID,isovalue, center,
                   zrange=(-inf,inf), n=24, exposed='above', elevation=None):
        """Return a radar representation of a field in the r3d

        Keyword arguments:
        fieldID   -- index or name of the field. If the variable can be
                     interpreted as a number, it is used as the index
                     (starting from 0) of the field in the r3d file.
                     Otherwise, it will be interpreted as a field name.
        isovalue  -- field value above which a cell is counted as
                     contaminated
        center    -- sequence with x and y coordinates of center
                     (release point)
        zrange    -- sequence with minimum and max z coordinates to check
                     (default (-inf,inf))
        n         -- number of sectors (default: 24)
        exposed   -- 'above' if values above the isovalue count as exposed
                     (e.g. gas concentration and radiation) and 'below' if
                     values below the isovalue count as exposed (e.g.
                     visible length, temperature of cold gas)
                     (default: 'above')
        elevation -- elevation object that has a method get_elevation(x,y)
                     which gives ground elevation for any x,y. When this
                     is given, the zrange is computed relative to the
                     ground level.
                     The elevation object should also have
                     a get_zrange() which returns a tuple
                     (lower_z, upper_z) which is a lower and an upper
                     bound for the elevation that may be returned.
                     get_zrange() may return (-inf,inf) or a large
                     interval, but with a tighter interval the
                     computation may be faster in the future.

        """
        grid = (self.x,self.y,self.z)

        # Get the correct data field from field index or name
        try:
            fieldindex = int(fieldID)
            data = self.fields[fieldindex]
        except ValueError:
            fieldname = str(fieldID)
            data = self.get_field_by_name(fieldname)

        # Pass on the call to the module function
        rmax_list = radar(data,grid,isovalue,center,zrange,n,exposed,elevation)
        return rmax_list

    def compute_gridlines(self,axis=None):
        """Return gridlines used to generate this r3d file"""
        if axis == 'x':
            gridlines = np.concatenate((self.xs[0:1], self.xs))
        elif axis == 'y':
            gridlines = np.concatenate((self.ys[0:1], self.ys))
        elif axis == 'z':
            gridlines = np.concatenate((self.zs[0:1], self.zs))
        elif axis is None:
            gridlines = [np.concatenate((self.xs[0:1], self.xs)),
                         np.concatenate((self.ys[0:1], self.ys)),
                         np.concatenate((self.zs[0:1], self.zs))]
        else:
            raise Error("The parameter may only be 'x', 'y', or 'z'")
        return gridlines

    def compute_cell_length(self,axis=None):
        """Return cell length of grid used to generate this r3d file"""
        if axis == 'x' or axis == 'y' or axis == 'z':
            gridlines = self.compute_gridlines(axis)
            cell_length_list = gridlines[1:] - gridlines[:-1]
        elif axis == None:
            cell_length_list = [self.compute_cell_length('x'),
                                self.compute_cell_length('y'),
                                self.compute_cell_length('z')]
        else:
            raise Error("The parameter may only be 'x', 'y', or 'z'")
        return cell_length_list

    def compute_cell_volume(self,xmin=small,xmax=big,ymin=small,ymax=big,zmin=small,zmax=big):
        """Return volume of control volumes inside box"""

        # Compute cell length
        dx_all,dy_all,dz_all = self.compute_cell_length()

        # Limit to cells within the specified box
        idx_inside = (xmin <= self.x) & (self.x <= xmax)
        jdx_inside = (ymin <= self.y) & (self.y <= ymax)
        kdx_inside = (zmin <= self.z) & (self.z <= zmax)
        dx = dx_all[idx_inside]
        dy = dy_all[jdx_inside]
        dz = dz_all[kdx_inside]

        # Compute volume of each cell within box
        dV = np.zeros((len(dx),len(dy),len(dz)))
        dA = np.outer(dx,dy)
        for k in range(len(dz)):
            dV[:,:,k] = dA*dz[k]

        return dV

    def write_vtk(self, filename):
        """Write the r3d object to a VTK (ASCII) file.

        If there are field named "U(m/s)", "V(m/s)", and "W(m/s)",
        they are interpreted as a vector field.
        All other fields are written as scalar fields.
        """

        with open(filename,'w') as f:
            f.write("# vtk DataFile Version 2.0\n")
            f.write("Kameleon FireEx KFX results file\n")
            f.write("ASCII\n")
            f.write("DATASET RECTILINEAR_GRID\n")

            f.write("DIMENSIONS %i %i %i \n" % self.shape)

            f.write("X_COORDINATES %i float\n" % self.shape[0])
            for x in self.x:
                f.write("%f " % x)

            f.write("\nY_COORDINATES %i float\n" % self.shape[1])
            for y in self.y:
                f.write("%f " % y)

            f.write("\nZ_COORDINATES %i float\n" % self.shape[2])
            for z in self.z:
                f.write("%f " % z)

            np = self.shape[0] * self.shape[1] * self.shape[2]
            f.write("\nPOINT_DATA %i" % np);

            # If we find a velocity field, write that as a vector field
            try:
                Uidx = self.names.index("U(m/s)")
                Vidx = self.names.index("V(m/s)")
                Widx = self.names.index("W(m/s)")
                write_velocity = True
            except ValueError:
                write_velocity = False

            scalar_names = self.names[:]

            if write_velocity:
                # Remove velocity fields from scalar_names
                scalar_names.remove("U(m/s)")
                scalar_names.remove("V(m/s)")
                scalar_names.remove("W(m/s)")

                # Write velocity to file
                f.write("\nVECTORS Velocity float\n")
                for k in range(self.shape[2]):
                    for j in range(self.shape[1]):
                        for i in range(self.shape[0]):
                            f.write("%f %f %f " % (self.fields[Uidx][i,j,k],
                                                   self.fields[Vidx][i,j,k],
                                                   self.fields[Widx][i,j,k]))

            # Write the rest of the variables as scalars
            for name in scalar_names:
                # Replace spaces with underscores in name
                f.write("\nSCALARS %s float\n" % name.replace (" ", "_"))
                f.write("LOOKUP_TABLE default\n");
                field = self.get_field_by_name(name)
                for k in range(self.shape[2]):
                    for j in range(self.shape[1]):
                        for i in range(self.shape[0]):
                            f.write("%f " % field[i,j,k])

    def to_type(self, newtype):
        """Converts the R3d object to the given type.

        Currently, only conversion to type 1 (from type 4) is
        supported.

        """
        # Raise an error if the current format is unknown
        if self.r3dtype not in (1,2,4):
            msgtemplate = "Cannot convert from unknown type {0}.\n"
            msg = msgtemplate.format(self.r3dtype)
            raise NotImplementedError(msg)

        if newtype == 1:
            if self.r3dtype == 1:
                return
            if self.r3dtype == 2:
                # We could explicitly unset coordinates[3:6],
                # but what is the point?
                # coordinates[0:3] will already have their correct
                # values.
                self.r3dtype = 1
            elif self.r3dtype == 4:
                self.coordinates[0] = staggered_to_center(self.coordinates[0])
                self.coordinates[1] = staggered_to_center(self.coordinates[1])
                self.coordinates[2] = staggered_to_center(self.coordinates[2])
                self.r3dtype = 1
        else:
            msg = "Only conversion to type 1 is currently supported"
            raise NotImplementedError(msg)



def readr3d(filename, gzipped=None, fieldnums=None):
    """ Read data from r3d file """

    from struct import unpack

    # Check whether the r3d file is gzipped or not if
    # not given
    if gzipped is None:
        with open(filename, 'rb') as f:
            magic = f.read(2)
            if magic == '\037\213':
                gzipped = True
            else:
                gzipped = False

    # Open gzipped or normal file
    if gzipped:
        f = gzip.open(filename, 'rb')
    else:
        f = open(filename, 'rb')

    # Read fixed size header
    raw_header = f.read(24)
    nx,ny,nz,nvar,r3dtype,t = unpack("iiiiif",raw_header)

    # Issue a warning if the r3dtype is not supported
    if r3dtype not in (1,2,4):
        rawmsg = "R3d file '{0}' is of type {1}, which is not supported."
        msg = rawmsg.format(filename, r3dtype)
        warn(msg)

    # Check if only a subset of the fields should be read
    if fieldnums is not None:
        # Make sure it is sorted
        fieldnum_list = list(fieldnums)
        fieldnum_list.sort()
    else:
        fieldnum_list = range(nvar)

    # Read the field names
    field_names_size = 32 * nvar
    field_names_format = '32s' * nvar
    rawnames = unpack(field_names_format,f.read(field_names_size))
    # Split off the zero byte and the following padding
    # Skip field names that should not be read (according to fieldnums)
    names = [ rawnames[i].split('\x00')[0] for i in fieldnum_list ]

    # Read coordinates
    # First and last elements are coordinates to boundary, rest are coordinates
    # for midpoint in cell.
    raw_coordinates =  f.read(4 * (nx + ny + nz))
    x = np.array(unpack(str(nx) + 'f', raw_coordinates[0:4*nx]))
    y = np.array(unpack(str(ny) + 'f', raw_coordinates[4*nx:4*(nx+ny)]))
    z = np.array(unpack(str(nz) + 'f', raw_coordinates[4*(nx+ny):4*(nx+ny+nz)]))

    # Read staggered coordinates if r3dtype == 2
    r3d_kwargs = {}
    if r3dtype == 2:
        staggered_coordinates =  f.read(4 * (nx + ny + nz))
        xs = np.array(unpack(str(nx) + 'f', staggered_coordinates[0:4*nx]))
        ys = np.array(unpack(str(ny) + 'f', staggered_coordinates[4*nx:4*(nx+ny)]))
        zs = np.array(unpack(str(nz) + 'f', staggered_coordinates[4*(nx+ny):4*(nx+ny+nz)]))
        r3d_kwargs['xs'] = xs
        r3d_kwargs['ys'] = ys
        r3d_kwargs['zs'] = zs

    # Read the 3D-arrays contained in the r3d file
    fields = []
    for i in range(nvar):
        if i in fieldnum_list:
            v = array(unpack(
                             str(nx*ny*nz) + 'f',
                             f.read(nx*ny*nz * 4)
                             ))
            # Without order='F', the order would be (nz,ny,nx)
            v = reshape(v, (nx,ny,nz), order='F')
            fields.append(v)
        else:
            # Skip this field
            f.seek(nx*ny*nz * 4, 1)

    # Close r3d file
    f.close()

    # Create R3d instance with the read values
    # Note that nx, ny, nz, and nvar are not passed to the constructor.
    # Those values are inferred from the dimension of the others
    r3d = R3d(x, y, z, names, fields, t, r3dtype, gzipped, **r3d_kwargs)
    return r3d


def radar(data, grid, isovalue, center,
            zrange=(-inf,inf), n=24, exposed='above', elevation=None):
    """Return list representing a radar diagram for an isosurface.

    This function does not need an R3d object, since the field
    data and grid coordinates are specified directly. The R3d
    method radar calls this function in order to calculate the
    radar diagram.

    """

    rmax_list = zeros(n)
    if exposed == 'above':
        contaminated_point_list = np.transpose(np.where(data > isovalue))
    elif exposed == 'below':
        contaminated_point_list = np.transpose(np.where(data < isovalue))


    # Compute list of dx and dy for each index
    dx_list = grid[0] - center[0]
    dy_list = grid[1] - center[1]

    # Loop through all contaminated points and find rmax in each sector
    for point in contaminated_point_list:
        z = grid[2][point[2]]

        # Skip point if outside zrange
        if elevation is None:
            # Test for zrange using absolute values
            if not (zrange[0] <= z <= zrange[1]):
                continue
        else:
            # Test for zrange relative to ground level
            x = grid[0][point[0]]
            y = grid[1][point[1]]
            ground_level = elevation.get_elevation(x,y)
            if not (zrange[0] <= z - ground_level <= zrange[1]):
                continue

        # Compute radius and angle for each point
        dx = dx_list[point[0]]
        dy = dy_list[point[1]]
        radius = np.sqrt(dx**2 + dy**2)
        theta = np.mod(np.pi/2. - np.arctan2(dy,dx),2.*np.pi)
        sector = int(theta/2./np.pi*n)
        # Update sector rmax if necessary
        if (radius > rmax_list[sector]):
            rmax_list[sector] = radius

    return rmax_list

def staggered_to_center(stag_array):
    """
    Return a list of cell centers given staggered list as in type 4 r3d.

    R3d files with r3dtype 4 store the coordinate on the positive side
    of the cell instead of the cell center. This is called "staggered".
    This function converts a staggered grid to a cell center grid under
    the assumption that the first grid cell has zero witdth (which is
    true for most r3d files, since the border cell has zero width).

    It seems to be the convention that r3d files that do not have
    zero-width border cells are coded as type 1.

    """
    gridline_array = np.concatenate((stag_array[0:1], stag_array))
    return (gridline_array[:-1] + gridline_array[1:])/2.

def gridlines_to_cellcenters(gridline_list):
    """Return a list of cell centers given a sequence of gridlines.

    The sequence of gridlines is assumed to be in increasing order.
    For compatibility with KFX and other kfxtools methods, the first
    two and last two gridline coordinates should be the same,
    representing a zero-width cell at the border.

    """
    return (gridline_list[:-1] + gridline_list[1:])/2.

def cellcenters_to_gridlines(center_list):
    """Return a list of gridlines given a sequence of cell centers.

    The sequence of cell centers are assumed to be in increasing order.
    The first and last cell center is assumed to come from a zero-width
    border cell. The returned gridlines therefore have two equal
    coordinates first and last, just like in the grid definition in
    a fsc-file.

    Due to rounding errors, the straightforward algorithm does not
    give equal border gridlines on both ends. Therefore, we compute the
    gridlines from both sides, then take a weighted average of the grid
    coordinates and finally set the border gridlines to their average.
    """
    gridline_list_left = np.zeros(len(center_list) + 1)
    gridline_list_right = np.zeros(len(center_list) + 1)
    gridline_list_left[0] = center_list[0]
    gridline_list_right[-1] = center_list[-1]
    for idx in range(len(center_list)):
        gridline_list_left[idx+1] = (
            gridline_list_left[idx] +
                2.*(center_list[idx] - gridline_list_left[idx]))
        gridline_list_right[-idx-2] = (
            gridline_list_right[-idx-1] +
                2.*(center_list[-idx-1] - gridline_list_right[-idx-1]))
    # Put all weight on left for first two coordinates (since they are equal by
    # definition, and on right for the last two (for the same reason).
    # Interpolate in between.
    weight = np.concatenate(([0.],np.linspace(0,1,len(center_list)-1),[1.]))
    gridline_list = (1-weight)*gridline_list_left + weight*gridline_list_right
    return gridline_list


def concentration_profile(data_field, free_volume_field,
                          limit_list=None, volume_list=None,
                          raw_output=False):
    """Return concentration profile for a field variable

    Arguments:
    data_field        -- The data field that typically is gas concentration
    free_volume_field -- free flow volume in each cell,
                         i.e. dx * dy * dz * vph.
    limit_list        -- Optional. List of field values (concentrations)
                         for which exposed volume is listed. The field
                         values should be listed in decreasing order.
                         A default list suitable for ExloRAM is used
                         if None is specified.
    volume_list       -- Optional. Split profile on these volumes instead
                         of on specified limits. The values are in
                         cubic metres and should be listed in increasing
                         order.
    raw_output        -- Optional boolean (default False). If set to True,
                         limit and volume lists are ignored and the full
                         sorted lists of fieldvalue vs. volume is returned.
                         The length is the same as the number of cells.

    """

    if limit_list is not None and volume_list is not None:
        raise Error('Only one of limit_list and volume_list keyword arguments may be specified')

    # Use ascending values internally, even though input and output are
    # descending. This is due to use of np.argsort() which is only able
    # to sort in accending order.

    # Use default concentration list if none is specified
    if limit_list is None:
        asc_limit_list = np.array(
                            [0., .001, .005, .010, .025, .05, .08,
                                                     .1, .25, .5, .8 ] +
                            range(1,101)
                            )
    else:
        # Make an accending view of the descending input
        asc_limit_list = limit_list[::-1]
        # The asc_limit_list should now be sorted in accending order
        # but sort it to make sure.
        asc_limit_list.sort()

    # Sort data and volume with data in ascending order
    data1d = data_field.ravel()
    volume1d = free_volume_field.ravel()
    asc_data_indices = np.argsort(data1d)
    asc_data = data1d[asc_data_indices]
    asc_volume = volume1d[asc_data_indices]

    # Compute the cumulative volume
    # This should be computed with highest value first, so input must be
    # descending (in field value, not volume)
    desc_cum_volume = np.cumsum(asc_volume[::-1])
    # Turn the cumulative volume back to ascending (in field value,
    # and therefore descending in cumulative volume)
    asc_cum_volume = desc_cum_volume[::-1]

    if raw_output:
        return asc_data[::-1],asc_cum_volume[::-1]
    elif volume_list is None:
        # Add a zero volume at the end for limits that are above max data value
        asc_cum_volume = np.concatenate((asc_cum_volume,np.zeros(1)))
        limits_and_data = np.concatenate((asc_limit_list, asc_data))
        # New positions of index i in limits_and_data when the list is sorted
        # (the first argsort() gives the old index of each position after sorting)
        new_positions = np.argsort(np.argsort(limits_and_data))
        # The first cell above the first limit is pushed up by only the first limit
        # (i.e. position of limit == old position of first cell above)
        # The first cell above the second limit is pushed up by the first two limits
        # (i.e. position of limit == old position of first cell above - 1)
        # The first cell above the nth limit is pushed up n positions
        # (i.e. position of limit == old position of first cell above - (n-1) )
        # Therefore subtracting 0,1,2,3,4,... from new_positions gives the
        # indices in asc_cum_volume with the total volume above corresponding limit.
        n = len(asc_limit_list)
        asc_vol_indices = new_positions[:n] - range(n)
        asc_vol_list = asc_cum_volume[asc_vol_indices]
        return asc_limit_list[::-1],asc_vol_list[::-1]
    else:
        # This part uses desc_cum_volume, where the field value descends as
        # the index grows, but the cumulative volume increases.
        # Use descending data as well:
        desc_data = asc_data[::-1]
        # Sort the volume_list (avoid problems if input isn't ascending)
        volume_list.sort()
        # This follows the same pattern as above, but now we insert volume
        # limits instead of field value limits.
        limits_and_volumes = np.concatenate((volume_list, desc_cum_volume))
        n = len(volume_list)
        new_positions = np.argsort(np.argsort(limits_and_volumes))[:n]
        desc_limit_indices = new_positions - range(n)
        # Limit the volume to the total volume
        desc_limit_indices = [min(i,len(desc_cum_volume)-1) for i in desc_limit_indices]
        desc_limit_list = desc_data[desc_limit_indices]
        desc_vol_list = desc_cum_volume[desc_limit_indices]
        return desc_limit_list,  desc_vol_list


# Depreciated function names (may still be in use by applications)
gridlines_to_cellcentres = gridlines_to_cellcenters
cellcentres_to_gridlines = cellcenters_to_gridlines

