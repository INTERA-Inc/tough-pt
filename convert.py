"""
visualize.py
----------------

This reads a TOUGH2 model and will generate a
mayavi scene for a particular parameter.


Kevin J. Smith
7/21/2016

Built for INT.C003.SWITZ

"""
import os
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
import pandas as pd
import pickle
import glob

from tough_input import Gener, Mesh, Incon
#from parse_mesh import Mesh
from tough_output import OutputFile

def mp7_to_mp5(basedir, modelname):

    mp7_filename = os.path.join(basedir, modelname+'.mppth')
    mp5_filename = os.path.join(basedir, modelname+'5.mppth')

    skiprows = 3
    dt_h = np.dtype([('SequenceNumber', np.int32),
                     ('Group', np.int32),
                     ('ParticleID', np.int32),
                     ('PathlinePointCount', np.int32)])

    dt7 = np.dtype([('CellNumber', np.int32),
                    ('GlobalX', np.float32),
                    ('GlobalY', np.float32),
                    ('GlobalZ', np.float32),
                    ('TrackinTime', np.float32),
                    ('LocalX', np.float32),
                    ('LocalY', np.float32),
                    ('LocalZ', np.float32),
                    ('Layer', np.int32),
                    ('StressPeriod', np.int32),
                    ('TimeStep', np.int32), ])

    dt5 = np.dtype([('ParticleID', np.int32),
                    ('GlobalX', np.float32),
                    ('GlobalZ', np.float32),
                    ('LocalZ', np.float32),
                    ('GlobalY', np.float32),
                    ('TrackinTime', np.float32),
                    ('j', np.int32),
                    ('i', np.int32),
                    ('k', np.int32),
                    ('TimeStep', np.int32), ])

    h_pth = np.loadtxt(mp7_filename, skiprows=skiprows, max_rows=1, dtype=dt_h)
    data_pth5 = np.array([], dtype=dt5)

    while h_pth.size > 0:

        skiprows += 1
        max_rows = h_pth['PathlinePointCount'].tolist()

        data_pth = np.loadtxt(mp7_filename,
                              skiprows=skiprows,
                              max_rows=max_rows,
                              dtype=dt7)

        db_pth5 = np.empty((max_rows,), dtype=dt5)
        db_pth5['ParticleID'] = h_pth['ParticleID']*np.ones(max_rows, dtype=np.int32)
        db_pth5['GlobalX'] = data_pth['GlobalX']
        db_pth5['GlobalY'] = data_pth['GlobalY']
        db_pth5['LocalZ'] = data_pth['LocalZ']
        db_pth5['GlobalZ'] = data_pth['GlobalZ']
        db_pth5['TrackinTime'] = data_pth['TrackinTime']
        db_pth5['j'] = data_pth['CellNumber']
        db_pth5['i'] = np.ones(max_rows, dtype=np.int32)
        db_pth5['k'] = data_pth['Layer']
        db_pth5['TimeStep'] = data_pth['StressPeriod']

        data_pth5 = np.append(data_pth5, db_pth5)

        skiprows += max_rows
        h_pth = np.loadtxt(mp7_filename, skiprows=skiprows, max_rows=1, dtype=dt_h)

    h_mp5 = '@ [ MODPATH 7.2.001 (TREF=   0.000000E+00 ) ]'
    fmt = ['%19u', '%27.15E', '%27.15E', '%19.7E', '%27.15E', '%27.15E',
           '%8u', '%8u', '%8u', '%9u']
    np.savetxt(mp5_filename, data_pth5, fmt=fmt, header=h_mp5, delimiter=' ')

    return None

def read_mpend(basedir, modelname, m_p = 1.0, write_to_file=True):

    import matplotlib.pyplot as plt

    EndFile = os.path.join(basedir, modelname + '.mpend')

    dt7 = np.dtype([('ParticleId', np.int32),
                    ('Status', np.int32),
                    ('FinalTrackingTime', np.float32),
                    ('InitialCellNumber', np.int32),
                    ('InitialLayer', np.int32),
                    ('InitialGlobalX', np.float32),
                    ('InitialGlobalY', np.float32),
                    ('InitialGlobalZ', np.float32),
                    ('FinalCellNumber', np.int32),
                    ('FinalLayer', np.int32),
                    ('FinalLocalZ', np.float32),
                    ('FinalGlobalX', np.float32),
                    ('FinalGlobalY', np.float32),
                    ('FinalGlobalZ', np.float32), ])

    data_end = np.loadtxt(EndFile,
                          skiprows=6,
                          usecols=[2,3,5,6,7,11,12,13,16,17,20,21,22,23],
                          dtype=dt7)

    idx1 = np.where(data_end['FinalLayer'] == 1)[0]
    idx2 = np.where(data_end['FinalLocalZ'] == 1.0)[0]
    idx = np.intersect1d(idx1, idx2)
    # t_final = np.insert(data_end['FinalTrackingTime'][idx], 0, 0.0) / (1000.0 * 365.25 * 24.0 * 3600.0 )
    t_final = np.insert(data_end['FinalTrackingTime'][idx], 0, 0.0)
    sec_to_ka = 1.0/(1000.0*365.25*24.0*3600.0)
    t_final = np.sort(t_final)
    # len_0 = len(t_final)
    # m_array = m_p*np.arange(len(t_final), dtype=np.float64)

    # t_final, t_unique = np.unique(t_final, return_index=True)
    # len_1 = len(t_final)
    # m_array = m_array[t_unique]
    # m_0 = 0.0 #205.0
    # m_array[m_array < m_0] = 0.0
    # m_array[m_array > m_0] -= m_0
    # t_final = t_final[m_array != 0.0]
    # m_array = m_array[m_array != 0.0]
    # m_array = np.insert(m_array, 0, 0.0)
    # t_final = np.insert(t_final, 0, 0.0)

    m_array = m_p*np.arange(len(t_final), dtype=np.float64)
    # md_array = np.gradient(m_array, t_final)

    # t_avg = []
    # md_avg = []

    # i_avg = 0
    # num_pts = 500
    # while i_avg < len(md_array):
    #     t_avg.append(np.mean(t_final[i_avg:i_avg+num_pts]))
    #     md_avg.append(np.mean(md_array[i_avg:i_avg+num_pts]))
    #     i_avg += num_pts

    plt.plot(t_final*sec_to_ka, m_array, 'k')
    #plt.plot(np.array(t_avg)*sec_to_ka, md_avg, 'k')
    plt.xlabel('Time [ka]')
    plt.ylabel('Mass of Particles Passing Top Boundary [kg]')
    #plt.ylabel('Mass Flow Rate (kg/sec)')
    #plt.axis([0.0,100.0,0.0,4.0e-7])
    plt.show()

    if write_to_file:
        DataFile = os.path.join(basedir, modelname + '.dat')
        hdr_text = 'First column is Time (sec), and second is total mass passing top boundary (kg)'
        np.savetxt(DataFile, np.transpose([t_final, m_array]), header=hdr_text)

    return None


def write_line(file, data, dtype=None):

    if dtype == None:
        if type(data) == str:
            # data is a string value
            file.write(data)
        elif type(data) == int:
            # data is an integer value
            np.array([data], dtype=np.int32).tofile(file, sep=' ')
        elif type(data) == float:
            # data is a floating point value
            np.array([data], dtype=np.float64).tofile(file, sep=' ')
        elif type(data) == list:
            # data is a list without dtype provided
            print('Data inputted is a list with no dtype.  Try again.')
            exit()
        else:
            # data type is unrecognized
            print('Data type for input is unrecognized.')
            exit()
    else:
        # dtype is provided
        if isinstance(dtype, list):
            # A list of dtypes is provided.  Go through each data value individually and write to file
            for dat,dt in zip(data, dtype):
                np.array([dat], dtype=dt).tofile(file, sep=' ')
                file.write(' ') # An additional space is necessary between entries
        else:
            # A single dtype is given
            if type(data) == np.ndarray:
                data.tofile(file, sep=' ')
            else:
                np.array(data, dtype=dtype).tofile(file, sep=' ')

    file.write('\n')
    return None

def get_layer_geom(i_elem, elements, connections):
    # Returns the top, bottom, and area of each element (i_elem) within a layer
    elem_subset = []
    for i in i_elem:
        elem_subset.append(elements[i])

    area_up = np.zeros(len(elem_subset))
    area_down = np.zeros(len(elem_subset))
    top = np.zeros(len(elem_subset))
    bot = np.zeros(len(elem_subset))
    for i_el, elem in enumerate(elem_subset):
        top[i_el] = elem.z
        bot[i_el] = elem.z
        max_dz = 0.0
        min_dz = 0.0
        dx = 0.0
        dy = 0.0
        is_n1_x = []
        is_n1_y = []
        num_x = 0
        num_y = 0

        for is_n1, i_conn, i_con_el in zip(elem.is_n1, elem.connections, elem.connected_elements):
            conn = connections[i_conn]
            con_el = elements[i_con_el]

            if conn.isot == 3:
                # Connection is vertical
                if is_n1:
                    # element is n1
                    dz = -conn.d1 * conn.betax
                else:
                    # element is n2
                    dz = conn.d2 * conn.betax
                if conn.betax != 0.0:
                    if elem.z + dz > top[i_el]:
                        top[i_el] = elem.z + dz
                        max_dz = dz
                    if elem.z + dz < bot[i_el]:
                        bot[i_el] = elem.z + dz
                        min_dz = dz
            else:
                # Connection is horizontal
                if conn.isot == 1:
                    # Connection is in x direction
                    # Check that no more than two terms are collected:
                    if num_x < 2:
                        d12 = conn.d1 if is_n1 else conn.d2
                        delta_x = abs(elem.x - con_el.x)
                        dx += delta_x*d12/(conn.d1 + conn.d2)
                        is_n1_x.append(is_n1)
                        num_x += 1
                elif conn.isot == 2:
                    # Connection is in y direction
                    # Check that no more than two terms are collected:
                    if num_y < 2:
                        d12 = conn.d1 if is_n1 else conn.d2
                        delta_y = abs(elem.y - con_el.y)
                        dy += delta_y*d12/(conn.d1 + conn.d2)
                        # dy += conn.d1 if is_n1 else conn.d2
                        is_n1_y.append(is_n1)
                        num_y += 1
                else:
                    print('Horizontal connection between ' + elem.name + ' and ' + con_el.name +
                          ' has no dx or dy')
                    print('BETAX: ' + str(conn.betax))
                    exit()
            if (conn.isot == 3) and ((is_n1 and conn.betax < 0.0) or
                                     (not is_n1 and conn.betax > 0.0)):
                # Connection is up
                area_up[i_el] += conn.areax
            elif (conn.isot == 3) and ((is_n1 and conn.betax > 0.0) or
                                       (not is_n1 and conn.betax < 0.0)):
                # Connection is down
                area_down[i_el] += conn.areax
            elif (conn.isot == 3) and (conn.betax == 0.0):
                # Connection is horizontal but has isot = 3
                print('Horizontal connection between ' + elem.name + ' and ' + con_el.name +
                      'has ISOT = 3')
                exit()

        if top[i_el] == elem.z:
            # Must be at top of model, so assume z coordinate is midway between cell vertical boundaries
            top[i_el] -= min_dz

        if bot[i_el] == elem.z:
            # Must be at bottom of model, so assume z coordinate is midway between cell vertical boundaries
            bot[i_el] -= max_dz

        dz = top[i_el] - bot[i_el]

        if num_x == 1: #not is_n1_x and (all(is_n1_x) or not any(is_n1_x)):
            # Element is on x boundary, so double dx calculation
            dx *= 2.0
        if num_y == 1: #not is_n1_y and (all(is_n1_y) or not any(is_n1_y)):
            # Element is on y boundary, so double dy calculation
            dy *= 2.0

        if dz != 0.0:
            # All cells are connected with a vertical neighbor
            if dx == 0.0 and dy != 0.0:
                dx = elem.volx / (dy * dz)
            elif dy == 0.0 and dx != 0.0:
                dy = elem.volx / (dx * dz)
            elif dx == 0.0 and dy == 0.0:
                # This is a 1D vertical grid
                dx = dy = np.sqrt(elem.volx / dz)
        else:
            # This is a horizontal grid
            if dx != 0.0 and dy != 0.0:
                dz = elem.volx / (dx * dy)
            elif dx != 0.0 and dy == 0.0:
                # This is a 1D grid in the x direction
                dy = dz = np.sqrt(elem.volx / dx)
            elif dx == 0.0 and dy != 0.0:
                # This is a 1D grid in the y direction
                dx = dz = np.sqrt(elem.volx / dy)
            else:
                print('All dimensions of element' + elem.name + 'are 0.0.')
                exit()

        setattr(elem, 'dx', dx)
        setattr(elem, 'dy', dy)
        setattr(elem, 'dz', dz)

    return top, bot, np.maximum(area_up, area_down)


def gen_layers(elements, connections):
    # Re-orders TOUGH ELEME listing (elements) based on vertical layers.  These are determined based on the ISOT and
    # BETAX values provided in the CONNE list (connections)

    els_unprocessed = np.ones(len(elements), dtype=bool)
    i_elem_prev = []  # A list of the elements processed in the previous pass
    inds_el = np.arange(len(elements), dtype=int)
    nndlay = -1

    while any(els_unprocessed):
        i_elem = []
        elem_x = []
        elem_y = []
        nstrt = nndlay + 1
        if all(els_unprocessed):
            # This is the first pass:
            for i_el in inds_el[els_unprocessed].tolist():
                elem = elements[i_el]
                is_top = True
                for i_conn, con_el, is_n1 in zip(elem.connections, elem.connected_elements, elem.is_n1):
                    conn = connections[i_conn]
                    if conn.isot == 3:
                        # Found a vertical connection
                        # Check if current element is on top of current connection:
                        # Either a) elem is n1 and betax > 0 (elem = n1 is on top)
                        #     or b) elem is not n1 and betax < 0 (elem = n2 is on top)
                        is_top = is_top and ((conn.betax >= 0.0 and is_n1) or
                                             (conn.betax < 0.0 and not (is_n1)))
                        if not is_top:
                            break
                if is_top:
                    # elem is on top layer of all remaining unprocessed elements
                    i_elem.append(i_el)
                    elem_x.append(elem.x)
                    elem_y.append(elem.y)
        else:
            # This is not the first pass
            for i_el in i_elem_prev:
                # Go through all elements connected to those selected in previous pass
                el_prev = elements[i_el]
                for i_el_c, i_conn, is_n1 in zip(el_prev.connected_elements, el_prev.connections, el_prev.is_n1):
                    # Find if connected element is at the bottom of the connection:
                    conn = connections[i_conn]
                    if (conn.isot == 3 and ((conn.betax >= 0.0 and is_n1) or
                                            (conn.betax < 0.0 and not (is_n1)))):
                        # Connected element is one below the previous top layer of remaining elements, so it is in the
                        # current top layer.

                        if i_el_c not in i_elem:
                            # Only add element if not already in the list
                            i_elem.append(i_el_c)
                            elem_x.append(elements[i_el_c].x)
                            elem_y.append(elements[i_el_c].y)
            if not i_elem:
                # No layers are left to add
                break

        els_unprocessed[i_elem] = False
        nndlay = nstrt + len(i_elem) - 1
        i_srt = np.lexsort((elem_y,elem_x))
        i_elem = np.array(i_elem, dtype=np.int64)[i_srt].tolist()
        i_elem_prev = i_elem
        top, bot, area = get_layer_geom(i_elem, elements, connections)

        yield Layer(i_elem, top, bot, area, nstrt, nndlay)


def get_mesh(mesh_file=None, force_read_raw=False): # LC 12/05/2020
    """ get the mesh; tries to load a pickled version first
    if force_read_raw, it will load the raw data; otherwise, it
    first attempts to read cached data (for quick import)
    """
    # mesh_file = os.path.join("data_eos7r_simu", "SMA_ZNO_2Dhr_gv1_pv1_gas")
    if mesh_file is None:
        mesh_file = os.path.join('data_3Deos5_simu', 'SMA_NL_3D_gv4_pv2_gas.mesh')
    pckfile = os.path.join(os.path.dirname(mesh_file), 'temp_mesh.pck')

    try:
        if force_read_raw:
            raise Exception
        mesh = Mesh.from_pickle(pckfile)
        print("Loading the mesh from pickle '{0}'".format(pckfile))
    except Exception as e:
        mesh = Mesh(mesh_file)
        mesh.to_pickle(pckfile)
        print("Pickled the input file")
    return mesh


def get_output(out_file=None, pckfile=None, force_read_raw=False): # LC 12/05/2020
    if out_file is None:
        # out_file = os.path.join("data_eos7r_simu", "SMA_ZNO_2Dhr_gv1_pv1_gas.out")
        # out_file = os.path.join("data_eos5_simu", "mikey.out")
        out_file = os.path.join('data_3Deos5_simu', 'OUTPUT_DATA')

    pckfile = os.path.join(os.path.dirname(out_file), 'temp_output.pck')

    try:
        if force_read_raw:
            raise Exception
        out = OutputFile.from_pickle(pckfile)
        print("Read state and transport data from pickle '{0}'".format(pckfile))
    except Exception as e:
        # raise(e)
        out = OutputFile(out_file)
        out.to_pickle(pckfile)
        print('Pickled the output file')
    return out


def get_gener(in_file = None, force_read_raw=False):

    if in_file is None:
        # in_file = os.path.join("data_eos7r_simu", "SMA_ZNO_2Dhr_gv1_pv1_gas")
        in_file = os.path.join('data_3Deos5_simu', 'SMA_NL_3D_gv4_pv2_gas.gener')

    return Gener.from_file(in_file)

def get_incon(in_file = None, force_read_raw=False):

    if in_file is None:
        # in_file = os.path.join("data_eos7r_simu", "SMA_ZNO_2Dhr_gv1_pv1_gas")
        in_file = os.path.join('data_3Deos5_simu', 'SMA_NL_3D_gv4_pv2_gas.incon')

    return Incon.from_file(in_file, extra_record=False)

def get_data(mesh_file=None, out_file=None, gener_file=None,
             incon_file=None, force_read_raw=False):
    """  get a list of pandas dataframes that are the union of state and mesh

    returns a tuple: a list of output times, a Mesh ElemeCollection object, an array of state (Element-based)
    output-data pandas dataframes, a Mesh ConneCollection object, and an array of transport (Connection-based)
    output-data pandas dataframes


    if force_read_raw, this will read the raw data from the TOUGH input and output files

    """

    # except Exception as e:
    mesh = get_mesh(mesh_file=mesh_file, force_read_raw=force_read_raw)
    out = get_output(out_file=out_file, force_read_raw=force_read_raw)
    gener = get_gener(in_file=gener_file, force_read_raw=force_read_raw)
    incon = get_incon(in_file=incon_file, force_read_raw=force_read_raw)

    elems = mesh.nodes
    connections = mesh.connections

    state_data = []
    transport_data = []
    gener_data = []
    t_unique, i_unique = np.unique(out.times, return_index=True) # MH edit, 5/13/2020
    # for ix, i in enumerate(out.hdr_locs[:-1]):
    for ix in i_unique: # MH edit, 5/13/2020
        print(ix)
        state_data.append(out.dataframe_for_step(ix))
        print('Done with state data!')
        transport_data.append(out.conn_dataframe_for_step(ix))
        print('Done with transport data!')
        # gener_data.append(out.gener_dataframe_for_step(ix))
        # print('Done with source/sink data!')

    # Output data are only provided for "unique" times, so repeat outputs are removed.  (MH edit, 5/13/2020)
    return t_unique, elems, state_data, connections, transport_data, gener_data, gener, incon

def update_transport_data(transport_df_data, mesh, velocity_data):

    el_names = velocity_data.iloc['Desc'].to_numpy(dtype='U5')
    connections = mesh.connections
    conne_data = connections.as_numpy_array()
    elements = mesh.nodes
    eleme_data = elements.as_numpy_array()

    for i_el, elem in enumerate(elements):
        elem = mesh.nodes[i_el]
        elem.connected_elements
        conne_data['isot'][elem.connections]
        conne_data['betax'][elem.connections]

    return transport_df_data

def get_interpolated_data(mesh_file=None, output_files=None, force_read_raw=False):

    """  get a list of pandas dataframes that are the union of state and mesh from interpolated data

    returns a tuple: a list of output times, a Mesh ElemeCollection object, an array of state (Element-based)
    output-data pandas dataframes, a Mesh ConneCollection object, and an array of transport (Connection-based)
    output-data pandas dataframes


    if force_read_raw, this will read the raw data from the TOUGH input and output files

    """

    # except Exception as e:
    mesh = get_mesh(mesh_file=mesh_file, force_read_raw=force_read_raw)
    elems = mesh.nodes
    connections = mesh.connections

    # fnames = glob.glob(os.path.join(output_files, 'r3*y'))
    times = []
    print(output_files)
    for fname in output_files:
        times.append(float(fname[fname.find('r3_')+3:-1]))

    i_srt = np.argsort(np.array(times))
    print(np.array(times)[i_srt])
    times = (np.array(times)[i_srt]*365.25*24.0*3600.0).tolist()

    num_els = len(elems)
    num_cons = len(connections)
    state_data = []
    transport_data = []
    gener_data = gener = incon = []
    eleme_data = elems.as_numpy_array()
    conne_data = connections.as_numpy_array()
    transport_df = pd.DataFrame({'ELEM1':conne_data['name1'],
                                 'ELEM2':conne_data['name2'],
                                 'INDEX':(np.arange(num_cons)+1).tolist(),
                                 'VEL(GAS)':np.zeros(num_cons),
                                 'VEL(LIQ.)':np.zeros(num_cons)})
    state_df = pd.DataFrame({'ELEM.':eleme_data['name'],
                             'INDEX':(np.arange(num_els)+1).tolist(),
                             'P':np.zeros(num_els),
                             'T':np.zeros(num_els),
                             'SG':np.zeros(num_els),
                             'XHYDG':np.zeros(num_els),
                             'XHYDL':np.zeros(num_els),
                             'DG':np.zeros(num_els),
                             'DL':np.zeros(num_els)})

    for ix in i_srt: # MH edit, 5/13/2020
        fname = output_files[ix]
        all_data = pd.read_csv(fname, delim_whitespace=True)
        state_df['ELEM'] = all_data['Desc']
        state_df.iloc[:,2:] = all_data.iloc[:,1:8]
        state_data.append(state_df)
        print('Done with state data!')

        vel_file_data = all_data.iloc[:,-6:].to_numpy()
        # Initialize velocity data in current data frame to 0.0:
        vel_data = np.zeros((num_cons,2))
        # Comb through all elements to update transport data:
        for i_el, elem in enumerate(elems):

            if i_el % 10000 == 0:
                print('On element ' + str(i_el+1) + ' of ' + str(num_els) + '.')

            # Store velocity data from file in easily accessible variables:
            vx_g = vel_file_data[i_el,0]
            vy_g = vel_file_data[i_el,1]
            vz_g = vel_file_data[i_el,2]
            vx_l = vel_file_data[i_el,3]
            vy_l = vel_file_data[i_el,4]
            vz_l = vel_file_data[i_el,5]

            # Comb through all connections of current element
            for i_con_el, i_conn, is_n1 in zip(elem.connected_elements, elem.connections, elem.is_n1):
                con_el = elems[i_con_el]
                conn_isot = connections[i_conn].isot

                # Update velocity data in current dataframe:
                if conn_isot == 1:
                    # Connection in x-direction
                    pos_x = con_el.x > elem.x
                    # Gas velocity
                    if vx_g != 0.0:
                        if (vx_g > 0.0 and pos_x) or (vx_g < 0.0 and not pos_x):
                            # Velocity in pos./neg. x direction and connected element is at higher/lower x coordinate.
                            vel_data[i_conn,0] = -vx_g if is_n1 else vx_g

                    # Liquid velocity:
                    if vx_l != 0.0:
                        if (vx_l > 0.0 and pos_x) or (vx_l < 0.0 and not pos_x):
                            # Velocity in pos./neg. x direction and connected element is at higher/lower x coordinate.
                            vel_data[i_conn,1] = -vx_l if is_n1 else vx_l

                elif conn_isot == 2:
                    # Connection in y-direction
                    pos_y = con_el.y > elem.y
                    # Gas velocity
                    if vy_g != 0.0:
                        if (vy_g > 0.0 and pos_y) or (vy_g < 0.0 and not pos_y):
                            # Velocity in pos./neg. y direction and connected element is at higher/lower y coordinate.
                            vel_data[i_conn,0] = -vy_g if is_n1 else vy_g
                    # Liquid velocity:
                    if vy_l != 0.0:
                        if (vy_l > 0.0 and pos_y) or (vy_l < 0.0 and not pos_y):
                            # Velocity in pos./neg. y direction and connected element is at higher/lower y coordinate.
                            vel_data[i_conn,1] = -vy_l if is_n1 else vy_l

                elif conn_isot == 3:
                    # Connection in z-direction
                    pos_z = con_el.z > elem.z
                    # Gas velocity:
                    if vz_g != 0.0:
                        if (vz_g > 0.0 and pos_z) or (vz_g < 0.0 and not pos_z):
                            # Velocity in pos./neg. z direction and connected element is at higher/lower z coordinate.
                            vel_data[i_conn, 0] = -vz_g if is_n1 else vz_g
                    # Liquid velocity:
                    if vz_l != 0.0:
                        if (vz_l > 0.0 and pos_z) or (vz_l < 0.0 and not pos_z):
                            # Velocity in pos./neg. y direction and connected element is at higher/lower y coordinate.
                            vel_data[i_conn, 1] = -vz_l if is_n1 else vz_l

        transport_df.iloc[:,3:] = vel_data
        transport_data.append(transport_df)
        print('Done with transport data!')
        # gener_data.append(out.gener_dataframe_for_step(ix))
        # print('Done with source/sink data!')

    # Output data are only provided for "unique" times, so repeat outputs are removed.  (MH edit, 5/13/2020)
    return times, elems, state_data, connections, transport_data, gener_data, gener, incon

def get_m(t, t_array, m_dot_array):
    # Given an array of times (t_array) and mass flow rates (m_dot_array), and assuming mass flow rate at
    # arbitry times is linearly interpolated, this routine returns the mass (m) at an arbitrary time (t)
    # provided by the user.

#     if t_array[0] != 0.0:
#         # If no mass flow is provided for t = 0, assume it is 0 at t = 0.
#         t_array = np.insert(t_array, 0, 0.0)
#         m_dot_array = np.insert(m_dot_array, 0, 0.0)

    if (type(t) == float) or (type(t) == np.float32) or (type(t) == np.float64):

        if t == t_array[0]:
            return 0.0

        i_max = np.where(t_array <= t)[0][-1]
        m = np.trapz(m_dot_array[0:i_max+1], x = t_array[0:i_max+1])

        if t > t_array[-1]:
            # The provided time (t) exceeds the last in the time array (t_array)
            m += (t - t_array[-1])*m_dot_array[-1]
        elif t != t_array[i_max]:
            # Add the interpolated portion of the mass if the inputted time (t) does not equal any in the time array
            # (t_array)
            m += (t - t_array[i_max])*(m_dot_array[i_max] + 0.5*(m_dot_array[i_max+1] - m_dot_array[i_max])*
                                                                (t - t_array[i_max])/(t_array[i_max+1] - t_array[i_max]))
    else:
        if type(t) == list:
            t = np.array(t)
        m = np.zeros(len(t))
        m0 = 0.0
        for i in np.arange(1,len(t_array)):
            m[t == t_array[i-1]] = m0
            ifx = np.logical_and(t > t_array[i-1], t < t_array[i])
            m[ifx] = m0 + (t[ifx] - t_array[i-1])*(m_dot_array[i-1] +
                                                   0.5*(m_dot_array[i] - m_dot_array[i-1])*
                                                       (t[ifx] - t_array[i-1])/(t_array[i] - t_array[i-1]))
            m0 = np.trapz(m_dot_array[0:i+1], x = t_array[0:i+1])
        m[t == t_array[-1]] = m0
        ifx = t > t_array[-1]
        m[ifx] = m0 + (t[ifx] - t_array[-1])*m_dot_array[-1]

    return m

def get_t_from_m(m, t_array, m_dot_array):
    # Given an array of times (t_array) and mass flow rates (m_dot_array), and assuming mass flow rate at
    # arbitrary times is linearly interpolated, this routine returns the time (t) that an arbitrary mass (m)
    # has accumulated.

    # if t_array[0] != 0.0:
    #     # If no mass flow is provided for t = 0, assume it is 0 at t = 0.
    #     t_array = np.insert(t_array,0,0.0)
    #     m_dot_array = np.insert(m_dot_array,0,0.0)

    if type(m) == float:

        if m == 0.0:
            return t_array[0]

        m_check = 0.0
        m_prev = 0.0
        i = 0

        while m_check < m and i < len(t_array):
            # Find the latest index of the m_dot array where accumulated mass exceeds the inputted mass (m).
            i += 1
            m_prev = m_check
            m_check = get_m(t_array[i], t_array, m_dot_array)

        if m > m_check:
            # The inputted mass (m) exceeds the total accumulated mass throughout the entire time array (t_array).
            # Assume the final mass flow rate (m_dot_array[-1]) remains constant after the final time (t_array[-1])
            t = t_array[-1] + (m - m_check)/m_dot_array[-1]
        elif m == m_check:
            # The inputted mass aligns with one of the times provided in the time array (t_array).
            t = t_array[i]
        else:
            j = i-1
            md_j = m_dot_array[j]
            t_j = t_array[j]
            a = (m_dot_array[j+1] - md_j)/(t_array[j+1] - t_j)
            t = t_array[j] + (-md_j + np.sqrt(md_j*md_j + 2.0*a*(m - m_prev)))/a

    else:
        m_array = get_m(t_array, t_array, m_dot_array)
        t = np.zeros(len(m))

        for i in np.arange(1,len(m_array)):
            ifx = np.logical_and(m > m_array[i-1], m < m_array[i])
            j = i - 1
            md_j = m_dot_array[j]
            t_j = t_array[j]
            if md_j == m_dot_array[i]:
                t[ifx] = t_j + (m[ifx] - m_array[j])/md_j
            else:
                a = (m_dot_array[i] - md_j) / (t_array[i] - t_j)
                t[ifx] = t_j + (-md_j + np.sqrt(md_j * md_j + 2.0 * a * (m[ifx] - m_array[j]))) / a
            t[m == m_array[i]] = t_array[i]

    return t


class Layer():
    #  represents a stacked layer in a TOUGH model
    def __init__(self, i_elem, top, bot, area, nstrt, nndlay):
        self.i_elem = i_elem  # Indices of elements in layer as ordered in TOUGH ELEME block
        self.top = top
        self.bot = bot
        self.area = area
        self.nstrt = nstrt
        self.nndlay = nndlay

    def __len__(self):
        return len(self.i_elem)


class LayerCollection():
    # represents all layers in a mesh
    def __init__(self, elements, connections):
        self.layers = []
        self.proc_layers(elements, connections)

    def proc_layers(self, elements, connections):
        max_nodelay = 0
        layers = []
        for i_lay, layer in enumerate(gen_layers(elements, connections)):
            layers.append(layer)
            if len(layer) > max_nodelay:
                max_nodelay = len(layer)
            print('Done with layer ' + str(i_lay+1) + '!')

        # Remove all layers not containing the largest number of elements or layers with elements
        # having large (>1.0E20) volumes.  These are boundary layers.
        layers_removed = 0
        print('Maximum number of elements per layer: ' + str(max_nodelay))
        for ilay, layer in enumerate(layers, start=1):
            vol_arr = []
            for i_el in layer.i_elem:
                vol_arr.append(elements[i_el].volx)
            if (len(layer) != max_nodelay or max(vol_arr) > 1.0E20):
                print('Removing layer ' + str(ilay) + '!')
                print('Number of elements in layer: ' + str(len(layer)))
                print('Maximum volume: ' + str(max(vol_arr)))
                layers_removed += 1
            else:
                layer.nstrt -= layers_removed
                layer.nndlay -= layers_removed
                self.layers.append(layer)

    @property
    def nodelay(self):
        nodelay = []
        for layer in self.layers:
            nodelay.append(len(layer))
        return nodelay

    @property
    def top(self):
        top = []
        for layer in self.layers:
            top.append(layer.top)
        return top

    @property
    def bot(self):
        bot = []
        for layer in self.layers:
            bot.append(layer.bot)
        return bot

    @property
    def area(self):
        area = []
        for layer in self.layers:
            area.append(layer.area)
        return area

    def __getitem__(self, item):
        return self.layers[item]

    def __len__(self):
        return len(self.layers)


class Data():
    """ represents all the data that we need to visualize """

    def __init__(self, times_list, elements, state_data, connections, transport_data, gener_data, geners, incon):
        self.times = times_list  # Times with state and transport data in TOUGH output file
        self.elements = elements  # Data from ELEME list (ElemeCollection object)
        self.state = state_data  # State data (primary variables) from each time
        self.connections = connections  # Data from CONNE list (ConneCollection object)
        self.transport = transport_data  # Transport data (flows) from each time
        self.gener = gener_data # Gener data (sources/sinks) from each time
        self.geners = geners
        self.incon = incon
        self.layers = LayerCollection(self.elements, self.connections)
        self.m2t = []
        self.t2m = []
        self.name2idx = dict()
        self.njag = 0
        self.initial_time = 0.0
        self.proc_elements()

    def proc_elements(self):

        # Re-order elements in Data class by MODFLOW convention (layer-by-layer)
        # self.elements = self.elements[self.m2t]
        elements = []
        len_t_elements = len(self.elements)
        m2t = np.array([], dtype=int)
        for layer in self.layers:
            for i_el in layer.i_elem:
                elements.append(self.elements[i_el])
            m2t = np.concatenate((m2t, layer.i_elem))

        self.elements.elements = elements
        # m2t returns a list of TOUGH elements ordered by MODFLOW layer convention
        self.m2t = m2t.tolist()

        t2m = -np.ones(len_t_elements, dtype=int)
        t2m[m2t] = np.arange(len(m2t), dtype=int)
        # t2m returns a list of MODFLOW elements ordered by TOUGH ELEME list.  All entries of t2m equalling -1
        # represent boundary cells
        self.t2m = t2m.tolist()
        i_srt = np.argsort(self.state[0]['INDEX'].to_numpy(dtype='int64'))
        for ind, isrt in enumerate(np.sort(self.state[0]['INDEX'].to_numpy(dtype='int64')), start=1):
            if ind != isrt:
                print('Missing element #' + str(ind), ' which is ' + self.elements[ind-1].name)
                exit()

        el_names = self.state[0]['ELEM.'].to_numpy(dtype='U5')[i_srt] #[m2t]
        el_names = el_names[m2t]
        el_names = np.char.zfill(np.char.replace(el_names,' ','0'),5)

        t2m_array = np.array(self.t2m)

        for i_el, elem in enumerate(self.elements):

            # Replace and re-order connected elements indices by MODFLOW convention
            con_els = t2m_array[elem.connected_elements]
            self.name2idx[el_names[i_el]] = i_el
            # Get indices of con_els of "active elements" (i.e., those not in a boundary layer)
            i_act = np.where(con_els != -1)[0]
            # From all connected elements, get indices of active cells, sorted in MODFLOW order:
            i_srt = i_act[np.argsort(con_els[i_act])]
            i_bnd = np.where(con_els == -1)[0].tolist()

            if i_bnd:
                # Add a list of connections elem has to boundary layers:
                setattr(elem, 'bound_connections', np.array(elem.connections)[i_bnd].tolist())
                # Add a is_n1 boolean list for connections to boundary layers:
                setattr(elem, 'bound_is_n1', np.array(elem.is_n1)[i_bnd].tolist())

            # Re-order connected elements by MODFLOW order and remove elements from boundary layers:
            elem.connected_elements = con_els[i_srt].tolist()
            # Re-order connection indices by convention above and remove entries from boundary layers::
            elem.connections = np.array(elem.connections)[i_srt].tolist()
            # Re-order is_n1 boolean list by convention above and remove entries from boundary layers::
            elem.is_n1 = np.array(elem.is_n1)[i_srt].tolist()

            self.njag += 1 + len(elem.connections)

        return None

    @property
    def ntimes(self):
        return len(self.times)

    # @property
    # def njag(self):
    #     return len(self.elements) + 2 * len(self.connections)

    @property
    def perlen(self):
        times = np.asarray(self.times) - self.initial_time
        len_stp = np.diff(times, append=times[-1])
        len_stp[-1] = 1.0
        return len_stp

    @classmethod
    def from_disk(cls, mesh_file=None, out_file=None, gener_file=None,
                       incon_file=None, force_read_raw=False):
        """ loads the data and returns an instance of the class.

        if force_read_raw is True, it will read the raw data and
        not first attempt to read it from a cached file
        """
        times, elems, state, conns, trans, gener, geners, incon = (
            get_data(mesh_file=mesh_file, out_file=out_file, gener_file=gener_file,
                     incon_file=incon_file, force_read_raw=force_read_raw))
        return cls(times, elems, state, conns, trans, gener, geners, incon)

    @classmethod
    def from_disk_interpolated(cls, mesh_file=None, output_files=None, force_read_raw=False):

        times, elems, state, conns, trans, gener, geners, incon = (
            get_interpolated_data(mesh_file = mesh_file, output_files = output_files,
                                  force_read_raw = force_read_raw))
        return cls(times, elems, state, conns, trans, gener, geners, incon)

    @classmethod
    def from_pickle(cls, file):
        """ return a mesh from a pickled file """
        with open(file, "rb") as f:
            return pickle.load(f)

    def to_pickle(self, file):
        """ dump to pickle """
        with open(file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def get_nlay_nodelay(self):
        z_coord = []
        for elem in self.elements:
            z_coord.append(elem.z)
        z_unique = np.unique(np.asarray(z_coord))

        nodelay = []
        for z_uni in z_unique:
            nodelay.append(len(np.where(z_coord == z_uni)[0]))

        return len(z_unique), nodelay

    def get_iac(self):
        # returns IAC array, which is a vector indicating the number of connections plus 1 for each node.
        iac = []
        for elem in self.elements:
            iac.append(len(elem.connected_elements) + 1)
        return np.asarray(iac, dtype=int)

    def get_ja(self):
        # Returns JA array, which is a list of cell number (n) followed by its connecting cell numbers (m)
        # for each of the m cells connected to cell n.
        ja = []
        for i_elem, elem in enumerate(self.elements):
            # Elements are ordered by layer (MODFLOW convention)
            ja.append(i_elem)
            for con_el in elem.connected_elements:
                ja.append(con_el)
        return np.asarray(ja)+1

    def get_ivc(self):
        # Returns index array indicating if connection is vertical
        ivc = []
        for i_elem, elem in enumerate(self.elements):
            ivc.append(False)
            for i_conn in elem.connections:
                ivc.append(self.connections[i_conn].isot == 3)
        return np.asarray(ivc, dtype=int)

    def get_cl12_fahl(self):
        # Returns an array of connection distances, in same order as JA list
        cl12 = []
        fahl = []
        for i_elem, elem in enumerate(self.elements):
            cl12.append(0.0)  # Zero distance from node to itself
            fahl.append(0.0)
            for is_n1, i_conn in zip(elem.is_n1, elem.connected_elements):
                conn = self.connections[i_conn]
                fahl.append(conn.areax)
                cl12.append(conn.d1 if is_n1 else conn.d2)
        return np.asarray(cl12, dtype=np.float64), np.asarray(fahl, dtype=np.float64)

    def build_disu(self, base_dir, modelname, num_stp=1):
        import flopy
        # try:
        #     model = flopy.modflow.Modflow.load(os.path.join(base_dir,modelname,'.nam'))
        #     disu = flopy.modflow.ModflowDisU.load(os.path.join(base_dir,modelname,'.disu'), model)
        #     return disu, model
        # except Exception as e:
        #     # raise(e)

        # Building a FloPy object to build DISU file (see https://modflowpy.github.io/flopydoc/mfdisu.html)
        model = flopy.modflow.Modflow(modelname=modelname, model_ws=base_dir, version='mfusg')  # MODFLOW object
        nodes = len(self.elements)  # Number of nodes
        nlay = len(self.layers)  # Number of layers
        nodelay = self.layers.nodelay
        njag = self.njag  # Inputting full matrix
        ivsd = 0  # Layer vertical subdiscretization (see note on nlay)
        nper = len(self.times)  # Number of stress periods
        itmuni = 1  # Time unit (seconds)
        lenuni = 2  # Length unit (meters)
        idsymrd = 0  # Indicates that full FV matrix will be provided (consider symmetry later)
        laycbd = 0  # Indicates no confining bed exists
        top = self.layers.top
        bot = self.layers.bot
        area = self.layers.area
        iac = self.get_iac()  # IAC array (see get_iac routine for more details)
        # njag = sum(iac)
        ja = self.get_ja()  # JA array (see get_ja routine for more details)
        ivc = self.get_ivc()  # IVC (indicates if connection is horizontal or vertical, see note on nlay)
        cl12, fahl = self.get_cl12_fahl()  # Returns array of connection distances and areas
        perlen = self.perlen  # Array of stress period lengths
        # perlen[-1] = 1.0e20
        steady = [False]*self.ntimes
        steady[-1] = True

        disu = flopy.modflow.mfdisu.ModflowDisU(model=model, nodes=nodes, nlay=nlay, njag=njag, ivsd=ivsd, nper=nper,
                                                itmuni=itmuni, lenuni=lenuni, idsymrd=idsymrd, laycbd=laycbd,
                                                nodelay=nodelay, area=area, top=top, bot=bot, iac=iac, ja=ja, ivc=ivc,
                                                cl12=cl12, fahl=fahl, perlen=perlen, steady=steady)

        model.write_input()

        return disu, model


    def build_gridmeta(self, model_ws, modelname, xorigin=0.0, yorigin=0.0, angrot=0.0):

        fname = os.path.join(model_ws, modelname + ".gridmeta")
        gmf = open(fname, 'w')
        write_line(gmf, "# Grid MetaData file for " + modelname + ".disu")
        write_line(gmf, [xorigin, yorigin, angrot], dtype=np.float64)
        ncells = len(self.elements)
        nlay = len(self.layers)
        write_line(gmf, [ncells, nlay], dtype=np.int32)
        ncpl = self.layers.nodelay
        write_line(gmf, ncpl, dtype=np.int32)
        dt = [np.int32, np.float32, np.float32, np.float32, np.float32]

        for i_el, elem in enumerate(self.elements, start=1):
            # write_line(gmf, [i_el, elem.x, elem.y, np.round_(elem.dx,decimals=2),
            #                  np.round_(elem.dy,decimals=2)], dtype=dt)
            gmf.write("{} {} {} {:.2f} {:.2f}\n".format(i_el, elem.x, elem.y, elem.dx, elem.dy)) # {:.2f} {:.2f}

            # if i_el > ncpl[0]:

            #     if np.round(elem.dx,2) != np.round(self.elements[i_el-1-ncpl[0]].dx,2):

            #         print('Element ' + elem.name + ' has a different dx than one layer above (Element ' +
            #               self.elements[i_el-1-ncpl[0]].name + ')')

            #     if np.round(elem.dy,2) != np.round(self.elements[i_el-1-ncpl[0]].dy,2):
            #         print('Element ' + elem.name + ' has a different dy than one layer above (Element ' +
            #               self.elements[i_el - 1 - ncpl[0]].name + ')')

        gmf.close()

        return None


    def build_hds(self, model_ws, modelname, hdry=1.0e30):

        fname = os.path.join(model_ws, modelname + ".hds")
        fbin = open(fname, 'wb')

        dt = np.dtype([('kstp', np.int32),
                       ('kper', np.int32),
                       ('pertim', np.float32),
                       ('totim', np.float32),
                       ('text', 'S16'),
                       ('nstrt', np.int32),
                       ('nndlay', np.int32),
                       ('ilay', np.int32), ])

        R_ch4 = 518.3
        R_h2 = 4124.0

        for kper in np.arange(self.ntimes):

            i_srt = np.argsort(self.state[kper]['INDEX'].to_numpy(dtype=np.int64))

            for ilay, layer in enumerate(self.layers):

                h = np.array((1,
                              kper + 1,
                              self.perlen[kper],
                              self.times[kper] - self.initial_time,
                              '           HEADU',
                              layer.nstrt + 1,
                              layer.nndlay + 1,
                              ilay + 1), dtype=dt)

                h.tofile(fbin)
                # kper += 1
                p_data = self.state[kper]['P'][i_srt][layer.i_elem]
                T_data = self.state[kper]['T'][i_srt][layer.i_elem]
                # rho_data = self.state[kper]['DW'][i_srt][layer.i_elem]
                sg_data = self.state[kper]['SG'][layer.i_elem]
                rho_data = self.state[kper]['DG'][layer.i_elem]
                rhol_data = self.state[kper]['DL'][layer.i_elem]
                # x_data = self.state[kper]['XWAT(1)'][i_srt][layer.i_elem]

                # rho_data = p_data/(R_h2*T_data)
                # sg_data = 1.0 - self.state[kper]['SL'][i_srt][layer.i_elem]
                x_data = self.state[kper]['XHYDG'][i_srt][layer.i_elem]
                # kper -= 1

                h_data = p_data/(9.81*rho_data)
                # h_data[x_data == 0.0] = hdry
                h_data[x_data == 0.0] = p_data/(9.81*rhol_data)
                h_data.to_numpy(dtype = np.float32).tofile(fbin)

        fbin.close()

        return None


    def build_hds2(self, model_ws, modelname, hdry=1.0e30, num_stp=10):

        fname = os.path.join(model_ws, modelname + ".hds")
        fbin = open(fname, 'wb')

        dt = np.dtype([('kstp', np.int32),
                       ('kper', np.int32),
                       ('pertim', np.float32),
                       ('totim', np.float32),
                       ('text', 'S16'),
                       ('nstrt', np.int32),
                       ('nndlay', np.int32),
                       ('ilay', np.int32), ])

        for kper in np.arange(self.ntimes):

            for kstp, frac in enumerate(np.linspace(1.0/num_stp, 1.0, num_stp), start=1):

                totim = self.times[kper] + frac * self.perlen[kper]

                for ilay, layer in enumerate(self.layers):

                    h = np.array((kstp,
                                  kper + 1,
                                  (frac - 1.0/num_stp)*self.perlen[kper], # self.perlen[kper]
                                  totim,
                                  '           HEADU',
                                  layer.nstrt + 1,
                                  layer.nndlay + 1,
                                  ilay + 1), dtype=dt)

                    h.tofile(fbin)
                    p_data = ((1.0-frac)*self.state[kper]['P'][layer.i_elem] +
                                    frac*self.state[kper+1]['P'][layer.i_elem])
                    rho_data = ((1.0-frac)*self.state[kper]['DG'][layer.i_elem] +
                                      frac*self.state[kper+1]['DG'][layer.i_elem])
                    sg_data = ((1.0-frac)*self.state[kper]['SG'][layer.i_elem] +
                                     frac*self.state[kper+1]['SG'][layer.i_elem])

                    h_data = p_data/(9.81*rho_data)
                    h_data[sg_data == 0.0] = hdry
                    h_data.to_numpy(dtype = np.float32).tofile(fbin)

        fbin.close()

    def build_budget(self, model_ws, modelname, area=None):

        fname = os.path.join(model_ws, modelname + ".cbc")
        fbin = open(fname, 'wb')

        dt = np.dtype([('kstp', np.int32),
                       ('kper', np.int32),
                       ('text', 'S16'),
                       ('njag', np.int32),
                       ('one', np.int32),
                       ('icode', np.int32),
                       ('imeth', np.int32),
                       ('delt', np.float32),
                       ('pertim', np.float32),
                       ('totim', np.float32), ])

        dt3 = np.dtype([('ndat', np.int32), ])
        dt4 = np.dtype([('auxtxt', 'S16'), ])
        h_well3 = np.array([(2, )], dtype=dt3)
        h_well4 = np.array([('           IFACE', )], dtype=dt4)
        dt5 = np.dtype([('nlist', np.int32)])
        dtw = np.dtype([('node', np.int32),
                        ('budget', np.float32),
                        ('iface', np.float32), ])

        if area == None:
            # Area is not provided.  Build a new area array.
            _, area = self.get_cl12_fahl()

        # phi_array = []
        # for i_el in self.m2t:
        #     phi_array.append(self.incon[i_el].porx)
        # phi_array = np.array(phi_array)

        R_ch4 = 518.3
        R_h2 = 4124.0

        el1_data = []
        el2_data = []
        el_list = list(self.name2idx.keys())
        bound_el_list = np.unique(self.state[0]['ELEM.'].to_numpy(dtype='U5')).tolist()
        for i_b, be in enumerate(bound_el_list):
            bound_el_list[i_b] = be.strip().zfill(5)
        bound_el_list = np.setdiff1d(bound_el_list, el_list)

        ind_ind_arr = self.transport[0]['INDEX'].to_numpy(dtype=np.int64)
        ind_uni, i_uni, ind_counts = np.unique(ind_ind_arr, return_index=True, return_counts=True)
        dupl_conn = ind_uni[np.where(ind_counts > 1)[0]]
        print('There are ' + str(len(dupl_conn)) + ' duplicated connections listed.  Here is a list...')
        for dc in dupl_conn:
            print(dc)
        # exit()
        i_srt = np.argsort(ind_uni)
        print('Sorting out which element is n1 and n2...')
        el1_arr = self.transport[0]['ELEM1'].to_numpy(dtype='U5')[i_uni]
        el2_arr = self.transport[0]['ELEM2'].to_numpy(dtype='U5')[i_uni]
        ind_arr = self.transport[0]['INDEX'].to_numpy(dtype='U6')[i_uni]

        num_conn = len(i_uni)

        # for el1, el2, ind in zip(self.transport[0]['ELEM1'].to_numpy(dtype='U5')[i_srt],
        #                          self.transport[0]['ELEM2'].to_numpy(dtype='U5')[i_srt],
        #                          self.transport[0]['INDEX'].to_numpy(dtype=np.int64)[i_srt]):

        for ind in np.arange(num_conn):

            el1 = el1_arr[ind]
            el2 = el2_arr[ind]
            el1_name = el1.strip().zfill(5)
            el2_name = el2.strip().zfill(5)

            try:
                el1_data.append(self.name2idx[el1_name])
                el2_data.append(self.name2idx[el2_name])
                continue

            except Exception as e:

                if el1_name in bound_el_list and el2_name in bound_el_list:
                    el1_data.append(0)
                    el2_data.append(0)
                    continue
                elif el1_name in bound_el_list:
                    el1_data.append(self.name2idx[el2_name])
                    el2_data.append(self.name2idx[el2_name])
                    continue
                elif el2_name in bound_el_list:
                    el1_data.append(self.name2idx[el1_name])
                    el2_data.append(self.name2idx[el1_name])
                    continue

            # else:
            #     el1_data.append(self.name2idx[el1_name])
            #     el2_data.append(self.name2idx[el2_name])

        for kper in np.arange(self.ntimes):

            print('Starting stress period ' + str(kper+1) + ' of ' + str(self.ntimes))

            # Cell to cell flow:
            h_ja = np.array((1,                                  # KSTP
                             kper + 1,                           # KPER
                             '    FLOW JA FACE',                 # TEXT
                             self.njag,                          # NVAL
                             1,                                  #
                             -1,                                 # ICODE (-1: compact style)
                             1,                                  # IMETH (full budget array)
                             self.perlen[kper],                  # DELT
                             0.0,                                # PERTIM
                             self.times[kper] - self.initial_time), dtype=dt)        # TOTIM

            h_st = np.array((1,                                  # KSTP
                             kper + 1,                           # KPER
                             '         STORAGE',                 # TEXT
                             len(self.elements),                 # NVAL
                             1,                                  #
                             -1,                                 # ICODE (-1: compact style)
                             1,                                  # IMETH (full budget array)
                             self.perlen[kper],                  # DELT
                             0.0,                                # PERTIM
                             self.times[kper] - self.initial_time), dtype=dt)        # TOTIM

            _, isrt_s = np.unique(self.state[kper]['INDEX'].to_numpy(dtype=np.int64), return_index=True)
            _, isrt_t = np.unique(self.transport[kper]['INDEX'].to_numpy(dtype=np.int64), return_index=True)
            vg_data = self.transport[kper]['VEL(GAS)'].to_numpy(dtype=np.float64)[isrt_t]
            vl_data = self.transport[kper]['VEL(LIQ.)'].to_numpy(dtype=np.float64)[isrt_t]
            # vl_data = self.transport[kper]['VEL(AQ.)'].to_numpy(dtype=np.float64)[isrt_t]
            # kper += 1
            # wt = 0.5
            # # vg_data = (wt*(self.transport[kper]['V(GAS)'].to_numpy(dtype=np.float64))
            # #            + (1.0 - wt)*(self.transport[kper+1]['V(GAS)'].to_numpy(dtype=np.float64)))
            # # vl_data = (wt*(self.transport[kper]['V(LIQ.)'].to_numpy(dtype=np.float64))
            # #            + (1.0 - wt)*(self.transport[kper+1]['V(LIQ.)'].to_numpy(dtype=np.float64)))
            # vg_data = (wt * (self.transport[kper]['VEL(GAS)'].to_numpy(dtype=np.float64))
            #            + (1.0 - wt) * (self.transport[kper + 1]['VEL(GAS)'].to_numpy(dtype=np.float64)))
            # vl_data = (wt * (self.transport[kper]['VEL(LIQ.)'].to_numpy(dtype=np.float64))
            #            + (1.0 - wt) * (self.transport[kper + 1]['VEL(LIQ.)'].to_numpy(dtype=np.float64)))
            ifl = np.where(vg_data == 0.0)[0]
            ifg = np.where(vg_data != 0.0)[0]
            v_data = np.empty(len(vg_data), dtype=np.float64)
            # v_data[ifg] = vg_data[ifg]
            v_data[ifg] = vg_data[ifg]
            v_data[ifl] = vl_data[ifl]

            xg_array = self.state[kper]['XHYDG'].to_numpy(dtype=np.float32)[isrt_s][self.m2t]
            xl_array = self.state[kper]['XHYDL'].to_numpy(dtype=np.float32)[isrt_s][self.m2t]
            # xl_array = self.state[kper]['XWAT(1)'].to_numpy(dtype=np.float32)[isrt_s][self.m2t]
            # xg_array = (wt * self.state[kper]['XRN2G'][self.m2t].to_numpy(dtype=np.float32)
            #             + (1.0 - wt) * self.state[kper + 1]['XRN2G'][self.m2t].to_numpy(dtype=np.float32))
            # xl_array = (wt * self.state[kper]['XRN2L'][self.m2t].to_numpy(dtype=np.float32)
            #             + (1.0 - wt) * self.state[kper + 1]['XRN2L'][self.m2t].to_numpy(dtype=np.float32))
            xg_array[np.argwhere(np.isnan(xg_array))] = 0.0
            xl_array[np.argwhere(np.isnan(xl_array))] = 0.0

            xg1_data = xg_array[el1_data]
            xg2_data = xg_array[el2_data]
            xl1_data = xl_array[el1_data]
            xl2_data = xl_array[el2_data]
            x1_data = np.empty(len(xg1_data), dtype=np.float64)
            x2_data = np.empty(len(xg2_data), dtype=np.float64)
            x1_data[ifg] = xg1_data[ifg]
            x1_data[ifl] = xl1_data[ifl]
            x2_data[ifg] = xg2_data[ifg]
            x2_data[ifl] = xl2_data[ifl]

            # x1_data = xl_array[el1_data]
            # x2_data = xl_array[el2_data]

            if_neg = np.where(v_data < 0.0)[0]
            if_pos = np.where(v_data >= 0.0)[0]

            v_data[if_neg] *= x1_data[if_neg]
            v_data[if_pos] *= x2_data[if_pos]

            # i_src_data = self.gener[kper]['ELEMENT'].to_numpy(dtype='U5')
            # i_min = 0
            # # i_min = int(0.5*len(i_src_data))  # MH, only for EOS7R simulation
            # i_src_data = i_src_data[i_min:]
            # # i_src_data = np.array(self.gener)
            # src_data = (wt*self.gener[kper]['GENERATION RATE'].to_numpy(dtype=np.float32) +
            #             (1.0-wt)*self.gener[kper+1]['GENERATION RATE'].to_numpy(dtype=np.float32))
            # src_data = src_data[i_min:]
            # i_min = int(0.5*len(i_src_data))  # MH, only for EOS7R simulation
            # # "den_data" is product of density, gas saturation, and porosity
            # phi_array = self.state[kper]['POROSITY'].to_numpy(dtype=np.float32)[isrt_s][self.m2t]
            # # sg_array_0 = self.state[kper]['SG'][self.m2t].to_numpy(dtype=np.float32)
            # # sg_array_1 = self.state[kper+1]['SG'][self.m2t].to_numpy(dtype=np.float32)
            # # rho_array_0 = self.state[kper]['DG'][self.m2t].to_numpy(dtype=np.float32)
            # # rho_array_1 = self.state[kper+1]['DG'][self.m2t].to_numpy(dtype=np.float32)
            # rho_array_0 = (self.state[kper]['P'][self.m2t].to_numpy(dtype=np.float32)/
            #                (R_ch4*(273.15+self.state[kper]['T'][self.m2t].to_numpy(dtype=np.float32))))
            # rho_array_1 = (self.state[kper+1]['P'][self.m2t].to_numpy(dtype=np.float32) /
            #                (R_ch4 * (273.15 + self.state[kper+1]['T'][self.m2t].to_numpy(dtype=np.float32))))
            # # rho_array_0[:i_min] *= R_ch4/R_h2
            # # rho_array_1[:i_min] *= R_ch4/R_h2
            # sg_array_0 = 1.0 - self.state[kper]['SL'][self.m2t].to_numpy(dtype=np.float32)
            # sg_array_1 = 1.0 - self.state[kper + 1]['SL'][self.m2t].to_numpy(dtype=np.float32)

            # den_data = rho_array*sg_array*phi_array

            # den_data = wt*(rho_array_0 * sg_array_0 * phi_array) + (1-wt)*(rho_array_1 * sg_array_1 * phi_array)

            ja_flow = np.zeros(self.njag, dtype = np.float32)
            stor = np.zeros(len(self.elements), dtype = np.float32)
            vmin = 1.0e-20
            ijag = 0
            # kper -= 1

            for i_el, elem in enumerate(self.elements):
                ja_flow[ijag] = 0.0
                ija = np.arange(ijag,ijag+len(elem.connections))+1
                vd = v_data[elem.connections]
                vd[np.logical_not(elem.is_n1)] *= -1.0
                # Multiply by mass fraction:
                # vd[abs(vd) < vmin] = vmin

                # Check if cell has only inflows or faces with no flow:
                if (np.all(vd >= 0.0) or np.all(vd <= 0)) and np.any(vd == 0.0):
                    # Change the first no-flow instance to slight negative value
                    vd[np.where(vd == 0.0)[0][0]] = (1.0 - 2.0*float(np.all(vd >= 0.0)))*vmin
                # Find cells with all inflows not occurring at the top or bottom boundaries:
                if ((np.all(vd > 0.0) or np.all(vd < 0.0)) and
                    ((i_el not in self.layers[0].i_elem) or
                     (i_el not in self.layers[-1].i_elem))):
                    # Find the location of the minimum velocity:
                    imin = np.argmin(vd)
                    # Change all velocities to nearly zero (vmin)
                    vd = (-1.0 + 2.0*float(np.all(vd > 0.0)))*vmin*np.ones(len(vd))
                    # Replace what was previously the minimum velocity to -vmin:
                    vd[imin] = (1.0 - 2.0*float(np.all(vd > 0.0)))*vmin

                ja_flow[ija] = vd*area[ija]
                stor[i_el] = -sum(ja_flow[ija])
                ijag += len(elem.connections) + 1

            h_well = np.array((1,                                  # KSTP
                               kper + 1,                           # KPER
                               '           WELLS',                 # TEXT
                               self.njag,                          # NVAL
                               1,                                  #
                               -1,                                 # ICODE (-1: compact style)
                               5,                                  # IMETH (list w/auxiliary data)
                               self.perlen[kper],                  # DELT
                               0.0,                                # PERTIM
                               self.times[kper] - self.initial_time), dtype=dt)        # TOTIM

            # Adding "WELLS" records for boundary connections and TOUGH sources/sinks
            well_data = np.array([], dtype=dtw)

            # # Simulate sources/sinks in TOUGH model as wells (sources/sinks at central nodes)
            # iface = 0.0
            # for src_el, src_dat in zip(i_src_data, src_data):
            #     i_src_el = self.name2idx[src_el.replace(' ','0').zfill(5)]
            #     den = den_data[i_src_el]
            #     if den > 0.0:
            #         # Add source/sink term to WELLS record
            #         q_data = src_dat/den
            #         well_data = np.append(well_data, np.array([(i_src_el+1, q_data, iface)], dtype=dtw), axis=0)
            #         stor[i_src_el] -= q_data

            # Simulate connections with boundary elements as wells (sources/sinks placed on interface locations)
            for i_el, elem in enumerate(self.elements, start=1):
                # if elem.bound_connections:
                if hasattr(elem, 'bound_connections'):
                    q_data = 0.0
                    v_datum = 0.0
                    for i_con, is_n1 in zip(elem.bound_connections,
                                            elem.bound_is_n1):
                        conn = self.connections[i_con]
                        v_datum = v_data[i_con] if is_n1 else -v_data[i_con]
                        q_data += (v_data[i_con] if is_n1 else -v_data[i_con])*conn.areax
                        iface = 5.0 # Assume flow is through bottom face
                        if conn.isot == 3:
                            if (is_n1 and conn.betax < 0.0) or (not is_n1 and conn.betax > 0.0):
                                # Flow is through top face
                                iface = 6.0
                        else:
                            print('Element '+ elem.name + ' has a non-vertical connection to a boundary.')
                            exit()
                        if q_data != 0.0:
                            well_data = np.append(well_data, np.array([(i_el, q_data, iface)], dtype=dtw), axis=0)
                            stor[i_el-1] -= q_data

            nlist = well_data.shape[0]
            h_well5 = np.array([(nlist, )], dtype=dt5)

            h_st.tofile(fbin)
            stor.tofile(fbin)
            h_ja.tofile(fbin)
            ja_flow.tofile(fbin)
            # h_well.tofile(fbin)
            # h_well3.tofile(fbin)
            # h_well4.tofile(fbin)
            # h_well5.tofile(fbin)
            # well_data.tofile(fbin)

        fbin.close()

        return None

    def build_budget2(self, model_ws, modelname, area=None, num_stp=10):

        fname = os.path.join(model_ws, modelname + ".cbc")
        fbin = open(fname, 'wb')

        dt = np.dtype([('kstp', np.int32),
                       ('kper', np.int32),
                       ('text', 'S16'),
                       ('njag', np.int32),
                       ('one', np.int32),
                       ('icode', np.int32),
                       ('imeth', np.int32),
                       ('delt', np.float32),
                       ('pertim', np.float32),
                       ('totim', np.float32), ])

        dt3 = np.dtype([('ndat', np.int32), ])
        dt4 = np.dtype([('auxtxt', 'S16'), ])
        h_well3 = np.array([(2, )], dtype=dt3)
        h_well4 = np.array([('           IFACE', )], dtype=dt4)
        dt5 = np.dtype([('nlist', np.int32)])
        dtw = np.dtype([('node', np.int32),
                        ('budget', np.float32),
                        ('iface', np.float32), ])

        if area == None:
            # Area is not provided.  Build a new area array.
            _, area = self.get_cl12_fahl()

        for kper in np.arange(self.ntimes)[:-1]:

            delt = self.perlen[kper] / (num_stp + 1)

            vg_0 = self.transport[kper]['V(GAS)'].to_numpy(dtype=np.float64)
            vg_1 = self.transport[kper+1]['V(GAS)'].to_numpy(dtype=np.float64)
            vl_0 = self.transport[kper]['V(LIQ.)'].to_numpy(dtype=np.float64)
            vl_1 = self.transport[kper+1]['V(LIQ.)'].to_numpy(dtype=np.float64)
            # vg_0 = self.transport[kper]['VEL(GAS)'].to_numpy(dtype=np.float64)
            # vg_1 = self.transport[kper + 1]['VEL(GAS)'].to_numpy(dtype=np.float64)
            # vl_0 = self.transport[kper]['VEL(LIQ.)'].to_numpy(dtype=np.float64)
            # vl_1 = self.transport[kper + 1]['VEL(LIQ.)'].to_numpy(dtype=np.float64)
            ifl_0 = np.where(vg_0 == 0.0)[0]
            ifg_0 = np.where(vg_0 != 0.0)[0]
            v0 = np.empty(len(vg_0), dtype=np.float64)
            v0[ifg_0] = vg_0[ifg_0]
            v0[ifl_0] = vl_0[ifl_0]
            ifl_1 = np.where(vg_1 == 0.0)[0]
            ifg_1 = np.where(vg_1 != 0.0)[0]
            v1 = np.empty(len(vg_1), dtype=np.float64)
            v1[ifg_1] = vg_1[ifg_1]
            v1[ifl_1] = vl_1[ifl_1]

            for kstp, frac in enumerate(np.linspace(1.0/num_stp, 1.0, num_stp), start=1):

                totim = self.times[kper] + frac*self.perlen[kper] - self.initial_time
                pertim = (frac - 1.0/num_stp)*self.perlen[kper]
                # Cell to cell flow:
                h_ja = np.array((kstp,                               # KSTP
                                 kper + 1,                           # KPER
                                 '    FLOW JA FACE',                 # TEXT
                                 self.njag,                          # NVAL
                                 1,                                  #
                                 -1,                                 # ICODE (-1: compact style)
                                 1,                                  # IMETH (full budget array)
                                 delt,                               # DELT
                                 pertim, #self.perlen[kper],         # PERTIM
                                 totim), dtype=dt)                   # TOTIM

                h_st = np.array((kstp,  # KSTP
                                 kper + 1,  # KPER
                                 '         STORAGE',  # TEXT
                                 len(self.elements),  # NVAL
                                 1,                   #
                                 -1,                  # ICODE (-1: compact style)
                                 1,                   # IMETH (full budget array)
                                 delt,                # DELT
                                 pertim, #self.perlen[kper],   # PERTIM
                                 totim), dtype=dt)    # TOTIM

                # vg_data = ((1.0-frac)*self.transport[kper]['V(GAS)'].to_numpy(dtype=np.float64) +
                #                  frac*self.transport[kper+1]['V(GAS)'].to_numpy(dtype=np.float64))
                # vl_data = ((1.0-frac)*self.transport[kper]['V(LIQ.)'].to_numpy(dtype=np.float64) +
                #                  frac*self.transport[kper]['V(LIQ.)'].to_numpy(dtype=np.float64))
                # ifl = np.where(vg_data == 0.0)[0]
                # ifg = np.where(vg_data != 0.0)[0]
                # v_data = np.empty(len(vg_data), dtype=np.float64)
                # v_data[ifg] = vg_data[ifg]
                # v_data[ifl] = vl_data[ifl]
                v_data = (1.0-frac)*v0 + frac*v1
                #v_data = vg_data
                i_src_data = self.gener[kper]['ELEMENT'].to_numpy(dtype='U5')
                # i_src_data = np.array(self.gener)
                src_data = ((1.0-frac)*self.gener[kper]['GENERATION RATE'].to_numpy(dtype=np.float32)
                                 +frac*self.gener[kper+1]['GENERATION RATE'].to_numpy(dtype=np.float32))
                # "den_data" is product of density, gas saturation, and porosity
                den_data = ((1.0-frac)*(self.state[kper]['DG'][self.m2t].to_numpy(dtype=np.float32) *
                                        self.state[kper]['SG'][self.m2t].to_numpy(dtype=np.float32) *
                                        self.state[kper]['POROSITY'][self.m2t].to_numpy(dtype=np.float32))
                                + frac*(self.state[kper+1]['DG'][self.m2t].to_numpy(dtype=np.float32) *
                                        self.state[kper+1]['SG'][self.m2t].to_numpy(dtype=np.float32) *
                                        self.state[kper+1]['POROSITY'][self.m2t].to_numpy(dtype=np.float32)))

                ja_flow = np.zeros(self.njag, dtype = np.float32)
                stor = np.zeros(len(self.elements), dtype = np.float32)
                vmin = 1.0e-30
                ijag = 0

                for i_el, elem in enumerate(self.elements):
                    ja_flow[ijag] = 0.0
                    ija = np.arange(ijag,ijag+len(elem.connections))+1
                    vd = v_data[elem.connections]
                    vd[np.logical_not(elem.is_n1)] *= -1.0

                    # Find cells with all inflows not occurring at the top or bottom boundaries:
                    if (np.all(vd > 0.0) and
                        ((i_el not in self.layers[0].i_elem) or
                         (i_el not in self.layers[-1].i_elem))):
                        # Find the location of the minimum velocity:
                        imin = np.argmin(vd)
                        # Change all velocities to nearly zero (vmin)
                        vd = vmin*np.ones(len(vd))
                        # Replace what was previously the minimum velocity to -vmin:
                        vd[imin] = -vmin
                    ja_flow[ija] = vd*area[ija]
                    stor[i_el] = -sum(ja_flow[ija])
                    ijag += len(elem.connections) + 1

                h_well = np.array((1,                                  # KSTP
                                   kper + 1,                           # KPER
                                   '           WELLS',                 # TEXT
                                   self.njag,                          # NVAL
                                   1,                                  #
                                   -1,                                 # ICODE (-1: compact style)
                                   5,                                  # IMETH (list w/auxiliary data)
                                   delt,                               # DELT
                                   pertim, #self.perlen[kper],         # PERTIM
                                   totim), dtype=dt)                   # TOTIM

                # Adding "WELLS" records for boundary connections and TOUGH sources/sinks
                well_data = np.array([], dtype=dtw)

                # Simulate sources/sinks in TOUGH model as wells (sources/sinks at central nodes)
                iface = 0.0
                for src_el, src_dat in zip(i_src_data, src_data):
                    i_src_el = self.name2idx[src_el.strip().zfill(5)]
                    den = den_data[i_src_el]
                    if den > 0.0:
                        # Add source/sink term to WELLS record
                        q_data = src_dat/den
                        well_data = np.append(well_data, np.array([(i_src_el+1, q_data, iface)], dtype=dtw), axis=0)
                        stor[i_src_el] -= q_data

                # Simulate connections with boundary elements as wells (sources/sinks placed on interface locations)
                for i_el, elem in enumerate(self.elements, start=1):
                    if elem.bound_connections:
                        q_data = 0.0
                        for i_con, is_n1 in zip(elem.bound_connections,
                                                elem.bound_is_n1):
                            conn = self.connections[i_con]
                            q_data += (v_data[i_con] if is_n1 else -v_data[i_con])*conn.areax
                            iface = 5.0 # Assume flow is through bottom face
                            if conn.isot == 3:
                                if (is_n1 and conn.betax < 0.0) or (not is_n1 and conn.betax > 0.0):
                                    # Flow is through top face
                                    iface = 6.0
                            else:
                                print('Element '+ elem.name + ' has a non-vertical connection to a boundary.')
                                exit()
                            if q_data != 0.0:
                                well_data = np.append(well_data, np.array([(i_el, q_data, iface)], dtype=dtw), axis=0)
                                stor[i_el-1] -= q_data

                nlist = well_data.shape[0]
                h_well5 = np.array([(nlist, )], dtype=dt5)

                h_st.tofile(fbin)
                stor.tofile(fbin)
                h_ja.tofile(fbin)
                ja_flow.tofile(fbin)
                h_well.tofile(fbin)
                h_well3.tofile(fbin)
                h_well4.tofile(fbin)
                h_well5.tofile(fbin)
                well_data.tofile(fbin)

        fbin.close()

        return None

    def build_mpbas(self, model_ws, modelname, hnoflo=1.0e20, hdry=1.0e30):

        fname = os.path.join(model_ws, modelname + '.mpbas')
        mpbf = open(fname, 'w')
        nlay = len(self.layers)

        header = '# MPBAS file created from ' + modelname + '.disu.'
        write_line(mpbf, header)
        write_line(mpbf, [hnoflo, hdry], dtype=np.float64)
        write_line(mpbf,1)
        write_line(mpbf,'WELLS')
        write_line(mpbf,5)
        laytyp = np.zeros(nlay, dtype=np.int32) # Convertible layer types
        write_line(mpbf, laytyp, dtype=np.int32)

        # Write IBOUND:
        for i_lay, layer in enumerate(self.layers, start=1):
            write_line(mpbf, 'INTERNAL 1 (' + str(len(layer.i_elem)) + 'I2) -1 # ibound layer ' + str(i_lay))
            ibound = np.ones(len(layer.i_elem), dtype=np.int32)
            write_line(mpbf, ibound, dtype=np.int32)

        # Set all porosities to 1.  Porosity effect already accounted for with velocity outputs.
        porosity = np.ones(len(self.elements), dtype = np.float64)
        for i_lay, layer in enumerate(self.layers, start=1):
            write_line(mpbf, 'INTERNAL 1 (' + str(len(layer.i_elem)) + 'E15.6) -1 # phi layer ' + str(i_lay))
            mpbf.write(' ')
            np.savetxt(mpbf, porosity[np.array(self.t2m)[layer.i_elem].tolist()], fmt='%14.6E', newline=' ')
            mpbf.write('\n')

        return None

    def build_mpsim(self, model_ws, modelname, SimulationType = 2):

        NameFileName = modelname + '.mpnam'
        ListingFileName = modelname + '.mplst'
        SimulationType = 2     # Pathline simulation
        TrackingDirection = 1  # Tracking Direction
        WeakSinkOption = 1     # Pass through weak sink cells
        WeakSourceOption = 1   # Pass through weak source cells
        BudgetOutputOption = 1 # Water balance errors not computed
        TraceMode = 1          # Trace mode (0: off, 1:on)
        EndpointFileName = modelname + '.mpend'
        PathlineFileName = modelname + '.mppth'
        TimeSeriesFileName = modelname + '.tim'
        TraceFileName = modelname + '.trc'
        TraceParticleGroup = 1
        TraceParticleId = 1
        BudgetCellCount = 0
        BudgetCellNumbers = np.array([], dtype = np.int32)
        ReferenceTimeOption = 1
        ReferenceTime = self.initial_time
        StressPeriod = 1
        TimeStep = 1
        TimeStepFraction = 0.0
        StopTimeOption = 1          # (1. Continue tracking until reaching end of flow simulation,
                                    #  2. Extend initial or final steady-state time steps and track all particles
                                    #     until they reach a termination location.)
        StopTime = self.times[-1]   # User-specified value of tracking time to stop
        TimePointOption = 1
        TimePointCount = 100
        TimePointInterval = 1.0e5 # Time interval between outputs (TimePointOption=1)
        TimePoints = np.array([], dtype=np.float64)  # Array of time values
        ZoneDataOption = 1      # Zone array data are not read
        StopZone = 0            # No automatic stop zone
        Zones = np.array([], dtype = np.int32)
        RetardationFactorOption = 1  # Retardation data no read
        Retardation = np.array([], dtype = np.float64)
        ParticleGroupCount = 1
        ParticleGroupName = '1'
        ReleaseOption = 1
        ReleaseTime = 0.0 # self.times[0]
        ReleaseTimeCount = 100
        InitialReleaseTime = 0.0 # self.times[0]
        ReleaseInterval = 1.0e6
        ReleaseTimes = np.linspace(self.times[0], StopTime, ReleaseTimeCount, dtype=np.float64)
        StartingLocationsFileOption = 'EXTERNAL'
        StartingLocationsFileName = modelname + '.loc'

        fname = os.path.join(model_ws, modelname + '.mpsim')
        simfile = open(fname, 'w')

        simfile.write('# MODPATH Simulation File\n')
        write_line(simfile, NameFileName)
        write_line(simfile, ListingFileName)
        write_line(simfile, [SimulationType, TrackingDirection, WeakSinkOption, WeakSourceOption,
                             BudgetOutputOption, TraceMode], dtype=np.uint8)
        write_line(simfile, EndpointFileName)
        if (SimulationType == 2) or (SimulationType == 4):
            write_line(simfile, PathlineFileName)
        if (SimulationType == 3) or (SimulationType == 4):
            write_line(simfile,TimeSeriesFileName)
        if (TraceMode == 1):
            write_line(simfile,TraceFileName)
            write_line(simfile,[TraceParticleGroup, TraceParticleId], dtype=np.int32)
        write_line(simfile, BudgetCellCount, dtype=np.int32)
        if BudgetCellCount > 0:
            write_line(simfile,BudgetCellNumbers)
        write_line(simfile, ReferenceTimeOption, dtype=np.int32)
        if ReferenceTimeOption == 1:
            write_line(simfile, ReferenceTime, dtype=np.float64)
        elif ReferenceTimeOption == 2:
            write_line(simfile, [StressPeriod, TimeStep, TimeStepFraction], dtype = np.float64)
        write_line(simfile, StopTimeOption, dtype=np.int32)
        if StopTimeOption == 3:
            write_line(simfile, StopTime, dtype=np.float64)
        if (SimulationType == 3) or (SimulationType == 4):
            write_line(simfile, TimePointOption, dtype=np.int32)
            if TimePointOption == 1:
                write_line(simfile, [TimePointCount, TimePointInterval], dtype=[np.int32, np.float64])
            elif TimePointOption == 2:
                write_line(simfile, TimePointCount, dtype=np.int32)
                write_line(simfile, TimePoints)
        write_line(simfile, ZoneDataOption, dtype=np.int32)
        if ZoneDataOption == 2:
            write_line(simfile, StopZone, dtype=np.int32)
            for layer in self.layers:
                write_line(simfile, layer.zones, dtype=np.int32)
        write_line(simfile, RetardationFactorOption, dtype=np.int32)
        if RetardationFactorOption == 2:
            for layer in self.layers:
                write_line(simfile, layer.retardation, dtype=np.int32)
        write_line(simfile, ParticleGroupCount, dtype=np.int32)
        write_line(simfile, ParticleGroupName)
        write_line(simfile, ReleaseOption, dtype=np.int32)
        if ReleaseOption == 1:
            write_line(simfile, ReleaseTime, dtype=np.float64)
        elif ReleaseOption == 2:
            write_line(simfile, [ReleaseTimeCount, InitialReleaseTime, ReleaseInterval],
                                dtype=[np.int32, np.float64, np.float64])
        elif ReleaseOption == 3:
            write_line(simfile, ReleaseTimeCount, dtype=np.int32)
            write_line(simfile, ReleaseTimes)
        simfile.write(StartingLocationsFileOption)
        if StartingLocationsFileOption == 'EXTERNAL':
            simfile.write(' ' + StartingLocationsFileName)
        else:
            print('You need to include Starting Locations Data here.')
            exit()

        simfile.close()

        return None

    def build_loc(self, model_ws, modelname):

        fname = os.path.join(model_ws, modelname + '.loc')
        loc_data = '# Particle initiation locations\n' # Header for file
        loc_data += '1\n' # InputStyle (locations specified)
        loc_data += '2\n' # LocationStyle (cell number)

        idx = []
        # el_names = self.gener[0]['ELEMENT'].to_numpy(dtype='U5')
        elem_data = self.elements.as_numpy_array()

        i_els = np.where(np.char.add(elem_data['ma1'], elem_data['ma2']) == 'SMACA')[0].tolist()
        el_names = elem_data['name'][i_els].tolist()
        # elem_names = []
        # for gener in elements[self.geners:
        #     el_names.append(gener.element)

        # all_els = self.state[0]['ELEM.'].to_numpy(dtype='U5')
        # el_names_chk = self.name2idx.keys()
        # bound_els = np.setdiff1d(all_els,el_names_chk)

        for el_name in el_names:
        #   print(el_name.strip().zfill(5))
            idx.append(self.name2idx[el_name.strip().zfill(5)])

        # idx.sort()
        idx = np.unique(np.array(idx)).tolist()
        t_drop = np.array([86.0]) * 365.25 * 24.0 * 3600.0

        ParticleCount = int(len(idx)*len(t_drop))
        loc_data += str(ParticleCount) + ' 1\n' # ParticleCount, ParticleIdOption (user-specified)
        ParticleId = 1

        for td in t_drop:

            for CellNumber in idx:

                loc_data += ("{} {} {} {} {} {} {}\n"
                    .format(
                    ParticleId,    # Particle ID (index)
                    CellNumber+1,  # Cell Number
                    0.5,           # Local X
                    0.5,           # Local Y
                    0.5,           # Local Z
                    td,            # Time Offset
                    0))            # Drape (Particles dropped in cell)

                ParticleId += 1

        loc = open(fname, 'w')
        loc.write(loc_data)
        loc.close()

        return None

    def build_loc_file(self, model_ws, modelname, locs_file, num_divs = 2):

        fname = os.path.join(model_ws, modelname + '.loc')
        loc_data = '# Particle initiation locations\n' # Header for file
        loc_data += '1\n' # InputStyle (locations specified)
        loc_data += '2\n' # LocationStyle (cell number)

        idx = []
        # el_names = self.gener[0]['ELEMENT'].to_numpy(dtype='U5')
        locs_file = os.path.join(os.path.dirname(model_ws), locs_file)
        el_names = np.loadtxt(locs_file, dtype='U5', delimiter=',', skiprows=1, usecols=0)

        for el_name in el_names:
            idx.append(self.name2idx[el_name.strip().zfill(5)])

        idx = np.unique(np.array(idx)).tolist()
        t_drop = np.array([0.0]) * 365.25 * 24.0 * 3600.0


        ParticleId = 0

        darr = 1.0/(2.0*num_divs) + np.arange(num_divs)/num_divs
        loc_data2 = ''

        for CellNumber in idx:

            for dx in darr:
                for dy in darr:
                    for dz in darr:

                        ParticleId += 1
                        loc_data2 += ("{} {} {} {} {} {} {}\n"
                            .format(
                            ParticleId,    # Particle ID (index)
                            CellNumber+1,  # Cell Number
                            dx,            # Local X
                            dy,            # Local Y
                            dz,            # Local Z
                            0.0,           # Time Offset
                            0))            # Drape (Particles dropped in cell)

        loc_data += str(ParticleId) + ' 1\n'
        loc_data += loc_data2

        loc = open(fname, 'w')
        loc.write(loc_data)
        loc.close()

        return None


    def build_loc2(self, model_ws, modelname, N_p, X_r = 1.0):

        # This routine drops particles at a rate dependent on the source terms
        # It inputs a total number of particles to drop N_p, and a mass fraction of
        # in the radionuclides from the hydrogen generated in the source terms

        fname = os.path.join(model_ws, modelname + '.loc')
        hdr_data = '# Particle initiation locations\n' # Header for file
        hdr_data += '1\n' # InputStyle (locations specified)
        hdr_data += '2\n' # LocationStyle (cell number)

        if type(X_r) == float:
            # Create an array of concentration values, all of which equal X_r:
            X_r *= np.ones(self.ntimes, dtype=np.float64)


        # sum_mp = sum(np.array(self.perlen)*self.gener['GENERATION RATE'].to_numpy(dtype=np.float64))
        sum_mp = 0.0
        for k_t, dt, Xr in zip(np.arange(self.ntimes), self.perlen, X_r):
            for gen_rate in self.gener[k_t]['GENERATION RATE'].to_numpy(dtype=np.float64):
                # The portion of the contaminant mass generated during the stress period k_t
                sum_mp += dt*(Xr*gen_rate)

        m_p = sum_mp/N_p # Mass of each particle (equally distributed)
        ParticleId = 0
        loc_data = ""
        for k_t, t, dt, Xr in zip(np.arange(self.ntimes), self.times, self.perlen, X_r):
            el_names = self.gener[k_t]['ELEMENT'].to_numpy(dtype='U5') # LC S5-S8 # MH S8-U5
            el_names = np.char.zfill(np.char.replace(el_names, ' ', '0'), 5)
            i_src = []
            for el_name in el_names:
                i_src.append(self.name2idx[el_name])
            # i_srt = np.argsort(i_src)
            # i_src = np.array(i_src)[i_srt].tolist()

            for gen_rate, CellNumber in zip(self.gener[k_t]['GENERATION RATE'].to_numpy(dtype=np.float64)[i_srt],
                                            i_src):
                
                drop_rate = Xr*gen_rate/m_p
                t_drop = np.linspace(t-dt, t, round(dt*drop_rate)+1.0)[:-1]
#               t_drop = np.linspace(t-dt, t, 5)[:-1]
                # print(len(t_drop))
                # t_drop = np.arange(t-dt, t, m_p/(Xr*gen_rate))
                # print(len(t_drop)); exit()

                for td in t_drop:
                    ParticleId += 1
                    loc_data += ("{} {} {} {} {} {} {}\n"
                        .format(
                        ParticleId,  # Particle ID (index)
                        CellNumber + 1,  # Cell Number
                        0.5,  # Local X
                        0.5,  # Local Y
                        0.5,  # Local Z
                        td,  # Time Offset
                        0))  # Drape (Particles dropped in cell)

        hdr_data += str(ParticleId) + ' 1\n'  # ParticleCount, ParticleIdOption (user-specified)

        loc = open(fname, 'w')
        loc.write(hdr_data)
        loc.write(loc_data)
        loc.close()

        # Update particle mass with exact number of particles dropped throughout the simulation:
        # m_p = sum_mp/ParticleId

        return m_p, ParticleId

    def build_loc3(self, model_ws, modelname, N_p, X_r=1.0, gen_type='COM4'):

        # This routine drops particles at a rate dependent on the source terms
        # It inputs a total number of particles to drop N_p, and a mass fraction of
        # in the radionuclides from the hydrogen generated in the source terms

        fname = os.path.join(model_ws, modelname + '.loc')
        hdr_data = '# Particle initiation locations\n'  # Header for file
        hdr_data += '1\n'  # InputStyle (locations specified)
        hdr_data += '2\n'  # LocationStyle (cell number)

        if type(X_r) == float:
            # Create an array of concentration values, all of which equal X_r:
            X_r *= np.ones(self.ntimes, dtype=np.float64)

        i_src = []
        total_m = []
        gens = []
        for gen in self.geners:
            if gen.type == gen_type:
                gens.append(gen)
                i_src.append(self.name2idx[gen.element])
                total_m.append(get_m(self.times[-1], gen.t_gen, gen.gx))

        m_p = sum(total_m) / N_p  # Mass of each particle (equally distributed)
        ParticleId = 0
        loc_data = ""
        for gen, CellNumber, tot_m in zip(gens, i_src, total_m):
            num_p = int(np.round(tot_m/m_p))
            md_array = np.linspace(0, tot_m, num_p+1)[1:]
            for td in get_t_from_m(md_array, gen.t_gen, gen.gx):
                ParticleId += 1
                loc_data += '{} {} {} {} {} {} {}\n'.format(ParticleId,  # Particle ID (index)
                                                            CellNumber + 1,  # Cell Number
                                                            0.5,  # Local X
                                                            0.5,  # Local Y
                                                            0.5,  # Local Z
                                                            td,   # Time Offset
                                                            0)    # Drape (Particles dropped in cell)

        hdr_data += str(ParticleId) + ' 1\n'  # ParticleCount, ParticleIdOption (user-specified)
        loc = open(fname, 'w')
        loc.write(hdr_data)
        loc.write(loc_data)
        loc.close()

        return m_p, ParticleId

if __name__ == '__main__':
    
    # pdir = os.path.join(".", "data_eos7r_simu") # LC 12/05/2020
    # base_dir = os.path.join(pdir, 'mp_run')
    # modelname = 'mp_SMA_ZNO_2Dhr_gv1_pv1_gas'
    # Information about input files:
    pdir = os.path.join('.', 'data_3Deos5_simu')
    data_pckfile = os.path.join(pdir, "datapickle.pck")
    base_dir = os.path.join(pdir, 'mp_run')
    base_file = 'SMA_NL_3D_gv4_pv2_gas'
    mesh_file = os.path.join(pdir, base_file + '.mesh')
    mesh_file = os.path.join(pdir, 'interpolated_data', 'SMA_NL_3D_prop_gv4_pv1_modpath_horiz_3m.eleme_conne')
    out_file = os.path.join(pdir, 'OUTPUT_DATA')
    out_files = glob.glob(os.path.join(pdir, 'interpolated_data', 'r3*y'))
    gener_file = os.path.join(pdir, base_file + '.gener')
    incon_file = os.path.join(pdir, base_file + '.incon')
    modelname = 'mp_' + base_file

    # mp7_to_mp5(base_dir, modelname); exit()

    try:
        data = Data.from_pickle(data_pckfile)
        print("Read from pickle {0}".format(data_pckfile))
    except Exception as e:
        # raise(e)
        # data = Data.from_disk(mesh_file=mesh_file, out_file=out_file, gener_file=gener_file, incon_file=incon_file)
        data = Data.from_disk_interpolated(mesh_file = mesh_file, output_files = out_files)
        data.to_pickle(data_pckfile)
        print('Pickled the data file')

    # data.geners = get_gener(in_file=gener_file)
    setattr(data, 'initial_time', 86.0*365.25*24.0*3600.0)

    hnoflo = 1.0e20
    hdry = 1.0e30
    N_p = 100000
#   X_r = 0.1*np.ones(data.ntimes) #mass fraction of the hydrogen
#   X_r[-3:] = 0.01
    X_r = 1.0  # MH Edit, 5/13/2020
    # # m_p, ParticleId = data.build_loc2(base_dir, modelname, N_p, X_r=X_r)
    num_stp = 1
    # m_p, ParticleId = data.build_loc3(base_dir, modelname, N_p, X_r=X_r, gen_type='COM4')
    # read_mpend(base_dir, modelname, m_p = m_p); exit()
    # print(m_p)
    # print(ParticleId)
    print('Writing DISU file...')
    disu, _ = data.build_disu(base_dir, modelname)
    print('Writing Grid Meta...')
    data.build_gridmeta(base_dir, modelname)
    print('Writing heads file for MODPATH...')
    data.build_hds(base_dir, modelname, hdry=0.5*hdry)
    print('Writing budget file for MODPATH...')
    data.build_budget(base_dir, modelname, area=None)
    # data.build_hds2(base_dir, modelname, hdry=0.5*hdry, num_stp=num_stp)
    # data.build_budget2(base_dir, modelname, area=disu.fahl, num_stp=num_stp)
    print('Writing locations file...')
    data.build_loc(base_dir, modelname)
    # locs_file = 'gv1_test_polygons_loc4.csv'
    # data.build_loc_file(base_dir, modelname, locs_file)
    print('Writing mpbas...')
    data.build_mpbas(base_dir, modelname, hdry=hdry, hnoflo=hnoflo)
    print('Writing mpsim...')
    data.build_mpsim(base_dir, modelname, SimulationType=2)
    #
    # Run the model:
    # os.system('mp7 ' + modelname + '.mpsim')

    exit()
