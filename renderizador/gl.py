#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    zBuffer = None

    def trans_matrix(x,y,z):
        return np.matrix([[1,0,0,x],
                        [0,1,0,y],
                        [0,0,1,z],
                        [0,0,0,1]])
    
    def rot_matrix(u,theta):
        cos = np.cos(theta/2)
        sin = np.sin(theta/2)

        # vector = np.array([0,0,1])
        # mag = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        # u = vector/mag


        qi = u[0] * sin
        qj = u[1] * sin
        qk = u[2] * sin
        qr = cos


        R = [[1-2*(qj**2+qk**2),2*(qi*qj - qk*qr)   , 2*(qi*qk + qj*qr)     ,0],
          [2*(qi*qj + qk*qr),1-2*(qi**2 + qk**2) , 2*(qj*qk - qi*qr)     ,0],
          [2*(qi*qk - qj*qr),2*(qj*qk+qi*qr)     ,1-2*(qi**2 + qj**2)    ,0],
          [0                ,0                   ,0                      ,1]]
        return R

    def scale_matrix(x,y,z):
        return np.array([[x,0,0,0],
                        [0,y,0,0],
                        [0,0,z,0],
                        [0,0,0,1]])
    
    def to_screen_matrix(width, height):
        return np.array([[width/2, 0, 0, width/2],
                        [0, -height/2, 0, height/2],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    def inter_area(x,y,x0,y0,x1,y1,x2,y2):
        A = 1/2 * abs((x0*(y1-y2)+x1*(y2-y0)+x2*(y0-y1)))
        A0 = 1/2 * abs((x*(y1-y2)+x1*(y2-y)+x2*(y-y1)))
        A1 = 1/2 * abs((x0*(y-y2)+x*(y2-y0)+x2*(y0-y)))
        A2 = 1/2 * abs((x0*(y1-y)+x1*(y-y0)+x*(y0-y1)))
        alpha = A0/A
        beta = A1/A
        gamma = A2/A

        return alpha, beta, gamma
    
    @staticmethod
    def mipmap_level(du_dx,du_dy,dv_dx,dv_dy):
        l = max(np.sqrt(du_dx**2 + dv_dx**2),np.sqrt(du_dy**2 + dv_dy**2))
        return int(np.log2(l))
    
    @staticmethod
    def generate_mipmap(image):
        mipmap_levels = [image]

        used_image = image
        while used_image.shape[0] > 1 and used_image.shape[1] > 1:
            
            height = max(1, used_image.shape[0] // 2)
            width = max(1, used_image.shape[1] // 2)

            
            reduced_image = np.zeros((height, width, used_image.shape[2]), dtype=used_image.dtype)

            for y in range(height):
                for x in range(width):
                    # Average the 2x2 block of pixels
                    block = used_image[2 * y:2 * y + 2, 2 * x:2 * x + 2]
                    reduced_image[y, x] = np.mean(block, axis=(0, 1))

            # Append the reduced image to the mipmap levels
            mipmap_levels.append(reduced_image)

            # Update used_image to the newly reduced image for the next iteration
            used_image = reduced_image

        return mipmap_levels

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        
        GL.cam_pos = [0, 0, 0]
        GL.cam_rot = [0, 0, 0]
        GL.stack = [np.identity(4)]
        GL.supersampling = np.zeros((GL.width*2, GL.height*2, 3))
        GL.zBuffer = - np.inf * np.ones((GL.width*2, GL.height*2))
        GL.directional_light = {"direction": np.array([0, 0, -1]), "color": np.array([1, 1, 1]), "intensity": 0}
        GL.start_time = time.time()

    @staticmethod
    def pushMatrix(matrix):
        """Empilhar a matriz atual."""
        GL.stack.append(GL.getMatrix()@matrix)

    @staticmethod
    def popMatrix():
        """Desempilhar a matriz atual."""
        GL.stack.pop()

    @staticmethod
    def getMatrix():
        """Obter a matriz atual."""
        return GL.stack[-1]


    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        emissiva = colors['emissiveColor']
        emissiva = [int(emissiva[0]*255), int(emissiva[1]*255), int(emissiva[2]*255)]

        for i in range(0, len(point), 2):
            pos_x = int(point[i])
            pos_y = int(point[i+1])
            print("Polypoint2D : ponto = {0}, {1}".format(pos_x, pos_y)) # imprime no terminal
            gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, emissiva)  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores

        emissiva = colors['emissiveColor']
        emissiva = [int(emissiva[0]*255), int(emissiva[1]*255), int(emissiva[2]*255)]
        n_lines = len(lineSegments) //2 - 1
        for i in range(0, n_lines):
            p0x = int(lineSegments[2*i])
            p0y = int(lineSegments[2*i+1])
            p1x = int(lineSegments[2*i+2])
            p1y = int(lineSegments[2*i+3])
            print("Polyline2D : ponto = {0}, {1}".format(p0x, p0y))

            dx = abs(p1x - p0x)
            dy = abs(p1y - p0y)

            if dx > dy:
                if p0x > p1x:
                    p0x, p0y, p1x, p1y = p1x, p1y, p0x, p0y
                for x in range(p0x, p1x):
                    y = p0y + (x - p0x)*(p1y - p0y)/(p1x - p0x)
                    if x>=0 and x<GL.width and y>=0 and y<GL.height:
                        gpu.GPU.draw_pixel([x, int(y)], gpu.GPU.RGB8, emissiva)
            else:
                if p0y > p1y:
                    p0x, p0y, p1x, p1y = p1x, p1y, p0x, p0y
                for y in range(p0y, p1y):
                    x = p0x + (y - p0y)*(p1x - p0x)/(p1y - p0y)
                    if x>=0 and x<GL.width and y>=0 and y<GL.height:
                        gpu.GPU.draw_pixel([int(x), y], gpu.GPU.RGB8, emissiva)

        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
        pos_x = GL.width//2
        pos_y = GL.height//2
        gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)


    @staticmethod
    def triangleSet2D(vertices, colors,
                    colorPerVertex = False,vertexColors = None ,z_values = None,
                    texture_values = None,image = None,
                    transparency = 0,
                    normals = None,
                    v = None):
        """Função usada para renderizar TriangleSet2D."""
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        if image is not None:
            mipmaps = GL.generate_mipmap(image)

        emissiva = colors['emissiveColor']
        emissiva = [int(emissiva[0]*255), int(emissiva[1]*255), int(emissiva[2]*255)]

        diffuse = colors['diffuseColor']
        diffuse = [int(diffuse[0]*255), int(diffuse[1]*255), int(diffuse[2]*255)]

        specular = colors['specularColor']
        specular = [int(specular[0]*255), int(specular[1]*255), int(specular[2]*255)]

        shininess = colors['shininess']
        
        lighting = GL.directional_light["intensity"]
        

        def L(p0, p1, ponto):

            vd = (p1[0]-p0[0],p1[1]-p0[1])
            vn = (vd[1],-vd[0])
            vp = (ponto[0]-p0[0],ponto[1]-p0[1])

            return(vp[0]*vn[0]+vp[1]*vn[1]) >= 0

        def inside(ponto_t0, ponto_t1,ponto_t2, ponto):
           return L(ponto_t0,ponto_t1,ponto) and L(ponto_t1,ponto_t2,ponto) and L(ponto_t2,ponto_t0,ponto)
           
        for i in range(0, int(len(vertices)/6)):
            x0 = vertices[i*6]
            y0 = vertices[i*6+1]
            x1 = vertices[i*6+2]
            y1 = vertices[i*6+3]
            x2 = vertices[i*6+4]
            y2 = vertices[i*6+5]


            area = 0.5*(x0*(y1-y2)+x1*(y2-y0)+x2*(y0-y1))
            if area>0:
                x0,y0,x1,y1,x2,y2 = x1,y1,x0,y0,x2,y2
    
            min_x = max(0, math.floor(min(x0, x1, x2)))
            max_x = min(GL.width - 1, math.ceil(max(x0, x1, x2)))
            min_y = max(0, math.floor(min(y0, y1, y2)))
            max_y = min(GL.height - 1, math.ceil(max(y0, y1, y2)))

            super_min_x = max(0, min_x*2)
            super_max_x = min(GL.width*2-1, max_x*2)
            super_min_y = max(0, min_y*2)
            super_max_y = min(GL.height*2-1, max_y*2)
            

            
            for x in range(super_min_x, super_max_x+1):
                for y in range(super_min_y, super_max_y+1):
                    if inside([x0*2,y0*2], [x1*2,y1*2], [x2*2,y2*2], [x+0.5,y+0.5]):
                        alpha, beta, gamma = GL.inter_area(x,y,x0*2,y0*2,x1*2,y1*2,x2*2,y2*2)


                        if z_values is not None:
                            z = 1/(alpha/z_values[0] + beta/z_values[1] + gamma/z_values[2])

                            if z > GL.zBuffer[x, y]:
                                GL.zBuffer[x, y] = z
                            else:
                                continue

                        if colorPerVertex:
                            r = alpha*vertexColors[3*i][0] + beta*vertexColors[3*i+1][0] + gamma*vertexColors[3*i+2][0]
                            g = alpha*vertexColors[3*i][1] + beta*vertexColors[3*i+1][1] + gamma*vertexColors[3*i+2][1]
                            b = alpha*vertexColors[3*i][2] + beta*vertexColors[3*i+1][2] + gamma*vertexColors[3*i+2][2]
                            cr = z * r/z_values[0]
                            cg = z * g/z_values[1]
                            cb = z * b/z_values[2]
                            color_used = [int(cr), int(cg), int(cb)]

                        elif texture_values is not None:
                            shape = image.shape[0]
                            u0 = texture_values[6*i]
                            v0 = texture_values[6*i+1]
                            u1 = texture_values[6*i+2]
                            v1 = texture_values[6*i+3]
                            u2 = texture_values[6*i+4]
                            v2 = texture_values[6*i+5]
                            u = alpha*u0 + beta*u1 + gamma*u2
                            v = alpha*v0 + beta*v1 + gamma*v2

                            # u e v vizinhos
                            alpha10, beta10, gamma10 = GL.inter_area(x+1, y, x0*2, y0*2, x1*2, y1*2, x2*2, y2*2)
                            u10 = alpha10*u0 + beta10*u1 + gamma10*u2
                            v10 = alpha10*v0 + beta10*v1 + gamma10*v2

                            alpha01, beta01, gamma01 = GL.inter_area(x, y+1, x0*2, y0*2, x1*2, y1*2, x2*2, y2*2)
                            u01 = alpha01*u0 + beta01*u1 + gamma01*u2
                            v01 = alpha01*v0 + beta01*v1 + gamma01*v2

                            du_dx = shape*(u10 - u)
                            dv_dx = shape*(v10 - v)
                            du_dy = shape*(u01 - u)
                            dv_dy = shape*(v01 - v)
                            d = GL.mipmap_level(du_dx,du_dy,dv_dx,dv_dy)

                            mipmap_used = mipmaps[d]
                            mipmap_shape = mipmap_used.shape[0]
                            
                            uz0 = u0/z_values[0]
                            vz0 = v0/z_values[0]
                            uz1 = u1/z_values[1]
                            vz1 = v1/z_values[1]
                            uz2 = u2/z_values[2]
                            vz2 = v2/z_values[2]

                            u_interpolated = (alpha * uz0 + beta * uz1 + gamma * uz2) / (alpha /z_values[0] + beta /z_values[1] + gamma /z_values[2])
                            v_interpolated = (alpha * vz0 + beta * vz1 + gamma * vz2) / (alpha /z_values[0] + beta /z_values[1] + gamma /z_values[2])

                            flipped_image = np.flip(mipmap_used[:, :, :3],axis=1)
                            r, g, b = flipped_image[min(255, int(u_interpolated * mipmap_shape)), min(255, int(v_interpolated * mipmap_shape))]

                            color_used = [min(255,int(r)), min(255,int(g)), min(255,int(b))]
                        else:
                            if not lighting:
                                color_used = emissiva
                            else:
                                #check if all 3 elements of normals are the same
                                if np.all(normals[0] == normals[1]) and np.all(normals[0] == normals[2]):
                                    normal = normals[0]
                                else:
                                    normal = np.array(alpha*normals[:,0] + beta*normals[:,1] + gamma*normals[:,2]).T[0]
                                    normal = normal / np.linalg.norm(normal)
                                cos_normal_light = np.dot(normal, GL.directional_light["direction"])
                                cos_normal_light = np.clip(cos_normal_light, 0, 1)

                                view_vector = alpha*v[0] + beta*v[1] + gamma*v[2]
                                view_vector = view_vector / np.linalg.norm(view_vector)

                                h = -GL.directional_light["direction"] + view_vector
                                h = h / np.linalg.norm(h)
                                
                                cos_normal_h = np.dot(-normal, h)
                                cos_normal_h = np.clip(cos_normal_h, 0, 1)
            
                                intensity = GL.directional_light["intensity"] * cos_normal_light
                                diffuse_lighting = intensity * np.array(diffuse)
                                
                                specular_lighting = GL.directional_light["intensity"]  * (cos_normal_h ** (shininess*128))
                                #print(f"specular_lighting: {specular_lighting}")
                                specular_lighting = [int(i) for i in specular_lighting * np.array(specular)]
                                

                                color_used = emissiva + diffuse_lighting + specular_lighting
                                #color_used = [int((i+1)*127.5) for i in normal]
                                #color_used = [int((i+1)*127.5) for i in h]
                                #color_used = [int(cos_normal_h*255), int(cos_normal_h*255), int(cos_normal_h*255)]
                                color_used = np.clip(color_used, 0, 255)

                        current_color = GL.supersampling[x][y]
                        color_used = [
                            int(color_used[0] * (1-transparency) + current_color[0] * transparency),
                            int(color_used[1] * (1-transparency) + current_color[1] * transparency),
                            int(color_used[2] * (1-transparency) + current_color[2] * transparency)
                        ]
                        
                        GL.supersampling[x][y] = color_used
            

            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                        p0 = GL.supersampling[x*2 - 1, y*2]
                        p1 = GL.supersampling[x*2, y*2]
                        p2 = GL.supersampling[x*2 - 1, y*2 + 1]
                        p3 = GL.supersampling[x*2, y*2 + 1]
                        mean = (p0 + p1 + p2 + p3) / 4
                        
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, mean)
                        
    @staticmethod
    def triangleSet(point, colors,colorPerVertex = False,vertexColors = None,
                    texture_values = None,image = None, normals = None):
        """Função usada para renderizar TriangleSet."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        transparencia = colors["transparency"]
        n_triangles = len(point) // 9
        tranform_matrix = GL.getMatrix()
        to_screen_matrix = GL.to_screen_matrix(GL.width, GL.height)
        lighting = GL.directional_light["intensity"]

        for i in range(0, n_triangles):
            p = point[i*9:i*9+9]
            x = p[0:9:3]
            y = p[1:9:3]
            z = p[2:9:3]
            tri_mat = np.array([x, y, z, [1, 1, 1]])
            
            tri_mat = tranform_matrix @ tri_mat
            operated_tri = GL.look_at @ tri_mat
            if normals is not None:
                normal = normals
                normal_0 = np.matrix(normal[9*i:9*i+3]).T
                normal_1 = np.matrix(normal[9*i+3:9*i+6]).T
                normal_2 = np.matrix(normal[9*i+6:9*i+9]).T

                normal_0 = np.vstack((normal_0, [0]))
                normal_1 = np.vstack((normal_1, [0]))
                normal_2 = np.vstack((normal_2, [0]))

                normal_matrix = np.hstack((normal_0, normal_1, normal_2))
                transformed_matrix = np.linalg.inv(GL.look_at @ tranform_matrix).T @ normal_matrix

                transformed_matrix = transformed_matrix[:3]

                normal = -transformed_matrix

            else:
                normal = np.cross(operated_tri[:3].transpose()[1] - operated_tri[:3].transpose()[0], operated_tri[:3].transpose()[2] - operated_tri[:3].transpose()[0])
                normal = -normal / np.linalg.norm(normal)
                normal = np.array([normal, normal, normal])

            z_values = operated_tri[2][0]
            tri_mat = GL.perspective_matrix @ tri_mat
            tri_mat = tri_mat / tri_mat[3][0]
            screen_matrix = to_screen_matrix @ tri_mat

            screen_matrix = np.array(screen_matrix)
            
            v1 = -1 * np.array(operated_tri[:3, 0]).flatten() / np.linalg.norm(operated_tri[:3, 0])
            v2 = -1 * np.array(operated_tri[:3, 1]).flatten() / np.linalg.norm(operated_tri[:3, 1])
            v3 = -1 * np.array(operated_tri[:3, 2]).flatten() / np.linalg.norm(operated_tri[:3, 2])
            v = [v1, v2, v3]

            GL.triangleSet2D(
                [
                    screen_matrix[0][0], screen_matrix[1][0],
                    screen_matrix[0][1], screen_matrix[1][1],
                    screen_matrix[0][2], screen_matrix[1][2]
                ], 
                colors,
                colorPerVertex,
                vertexColors[3*i:3*i+3] if colorPerVertex else None,
                np.array(z_values)[0],
                texture_values[6*i:6*i+6] if texture_values is not None else None,
                image = image,
                transparency = transparencia,
                normals = normal,
                v = v if lighting else None
                )

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.
        fovy =  2 * np.arctan(np.tan(fieldOfView/2) * GL.height / np.sqrt(GL.height**2 + GL.width**2))
        aspect_ratio = GL.width / GL.height
        near = GL.near
        far = GL.far
        top = near * np.tan(fovy)
        right = top * aspect_ratio

        cam_trans = np.linalg.inv(GL.trans_matrix(position[0], position[1], position[2]))
        cam_rot  = np.linalg.inv(GL.rot_matrix(orientation[:3], orientation[3]))
        look_at = cam_rot @ cam_trans
        GL.look_at = look_at
                
        perspective_matrix = np.array([[near/right, 0, 0, 0],
                                       [0, near/top, 0, 0],
                                       [0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
                                       [0, 0, -1, 0]])

        GL.perspective_matrix = perspective_matrix @ look_at

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        GL.translation = [0, 0, 0]
        GL.scale = [1, 1, 1]
        GL.rotation = [0, 0, 0, 0]
        
        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print("Transform : ", end='')
        if translation is not None:
            GL.translation = np.array(translation).flatten()
            #print("translation = {0} ".format(translation), end='') # imprime no terminal
        if scale:
            GL.scale = scale
            #print("scale = {0} ".format(scale), end='') # imprime no terminal
        if rotation is not None:
            GL.rotation = np.array(rotation).flatten()
            #print("rotation = {0} ".format(rotation), end='') # imprime no terminal
        translation_matrix = GL.trans_matrix(GL.translation[0], GL.translation[1], GL.translation[2])
        rotation_matrix = GL.rot_matrix(GL.rotation[:3], GL.rotation[3])
        scale_matrix = GL.scale_matrix(GL.scale[0], GL.scale[1], GL.scale[2])
        
        tranform_matrix = translation_matrix @ rotation_matrix @ scale_matrix
        GL.pushMatrix(tranform_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.
        GL.popMatrix()


    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        for i in range(0, len(point)-6, 3):
            v1 = point[i:i+3]
            v2 = point[i+3:i+6]
            v3 = point[i+6:i+9]
            GL.triangleSet([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]], colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors,
                                colorPerVertex=False, vertexColors=None, colorIndex=None,
                                texCoord=None, texCoordIndex=None, image=None):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        verts = []
        color_list = []
        texture_list = []
        i = 0 
        while i < len(index) - 2: 
            if index[i] == -1 or index[i + 1] == -1 or index[i + 2] == -1:
                i += 1
                continue  # Para a execução se encontrar -1 no índice
            
            # Pega as coordenadas dos vértices usando os índices da lista 'index'
            v1 = point[3 * index[i] : 3 * index[i] + 3]
            v2 = point[3 * index[i + 1] : 3 * index[i + 1] + 3]
            v3 = point[3 * index[i + 2] : 3 * index[i + 2] + 3]

            verts.extend(v1)
            verts.extend(v2)
            verts.extend(v3)

            if colorPerVertex and colorIndex is not None:
                c1 = colorIndex[i] * 3
                c2 = colorIndex[i + 1] * 3
                c3 = colorIndex[i + 2] * 3
                color1 = vertexColors[c1 : c1 + 3]
                color2 = vertexColors[c2 : c2 + 3]
                color3 = vertexColors[c3 : c3 + 3]
                color_list.extend(color1)
                color_list.extend(color2)
                color_list.extend(color3)

            elif texCoord is not None:
                t1 = texCoordIndex[i] * 2
                t2 = texCoordIndex[i + 1] * 2
                t3 = texCoordIndex[i + 2] * 2
                tex1 = texCoord[t1 : t1 + 2]
                tex2 = texCoord[t2 : t2 + 2]
                tex3 = texCoord[t3 : t3 + 2]
                texture_list.extend(tex1)
                texture_list.extend(tex2)
                texture_list.extend(tex3)
            
            GL.triangleSet(verts, colors,
                    colorPerVertex, vertexColors=color_list if colorPerVertex else None,
                    texture_values = texture_list if image is not None else None,image = image)
            
            # Avança para o próximo conjunto de vértices
            i += 1

    @staticmethod
    def indexedFaceSet(
        coord,
        coordIndex,
        colorPerVertex,
        color,
        colorIndex,
        texCoord,
        texCoordIndex,
        colors,
        current_texture,
    ):
        """Função usada para renderizar IndexedFaceSet."""
        
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
    
        vert_colors = None
        

        
        if colorPerVertex and color and colorIndex:
            
            vert_colors = []
            for idx in colorIndex:
                if idx != -1:
                    rgb = color[idx*3:idx * 3 + 3]
                    
                    vert_colors.append([int(c * 255) for c in rgb])
                    
        faces = []
        vertices = []
    
        for i in coordIndex:
            if i == -1:
                if vertices:
                    faces.append(vertices)
                vertices = []
            else:
                vertices.append(i)
        
        strips = []

        for face in faces:
            if len(face) < 3:
                continue
            
            for i in range(1, len(face) - 1):
                strips.append([face[0], face[i], face[i + 1], -1])

        strips_flat = []
        for sublist in strips:
            for item in sublist:
                strips_flat.append(item)

        GL.indexedTriangleStripSet(coord, strips_flat,colors,
                                colorPerVertex and color and colorIndex, vert_colors if vert_colors else colors,colorIndex,
                                texCoord, texCoordIndex, image if current_texture else None) 
    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.
        sx, sy, sz = size
        sx, sy, sz = sx/2, sy/2, sz/2
        verts = [
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz]
        ]

        triangles = [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [3, 2, 6],
            [3, 6, 7],
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 2],
            [2, 5, 6],
            [0, 3, 7],
            [0, 7, 4]
        ]
        for t in triangles:
            v1 = verts[t[0]]
            v2 = verts[t[1]]
            v3 = verts[t[2]]
            GL.triangleSet([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]], colors)




    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        
        div_long = 15
        div_lat = 15

        vertices = []
        delta_theta = 2 * np.pi / div_long
        delta_phi = np.pi / div_lat
        triangulos = []
        normals = []
        t_normals = []
        
        for i in range(div_long+1):
            theta = i * delta_theta
            for j in range(1,div_lat):
                phi = j * delta_phi
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                vertices.append([x, y, z])
                normals.append(np.array([x, y, z])/np.linalg.norm([x, y, z]))

        vertices.append([0, 0, radius])
        normals.append(np.array([0, 0, 1]))

        vertices.append([0, 0, -radius])
        normals.append(np.array([0, 0, -1]))

        polo_norte = len(vertices) - 2
        polo_sul = len(vertices) - 1
        
        for i in range(div_long):
            for j in range(div_lat-2):
                p1 = i * (div_lat-1) + j
                p2 = p1 +1
                p3 = (i + 1) * (div_lat-1) + j
                p4 = p3 + 1

                v1 = vertices[p1]
                v2 = vertices[p2]
                v3 = vertices[p3]
                v4 = vertices[p4]

                triangulos.extend([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]])
                t_normals.append([normals[p1],normals[p2],normals[p3]])
                v2, v3, v4 = v3, v2, v4
                triangulos.extend([v2[0], v2[1], v2[2], v3[0], v3[1], v3[2], v4[0], v4[1], v4[2]])
                t_normals.append([normals[p3],normals[p2],normals[p4]])
        
            v1 = vertices[polo_norte] 
            v2 = vertices[i * (div_lat-1)]
            v3 = vertices[(i + 1) % div_long* (div_lat-1)]

            triangulos.extend([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]])
            t_normals.append([
                normals[polo_norte],
                normals[i * (div_lat-1)],
                normals[(i + 1) % div_long* (div_lat-1)]
                ])


            v1 = vertices[polo_sul]
            v2 = vertices[(i + 1) * (div_lat-1) + div_lat-2]
            v3 = vertices[i * (div_lat-1) + div_lat-2]

            triangulos.extend([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]])
            t_normals.append([
                normals[polo_sul],
                normals[(i + 1) * (div_lat-1) + div_lat-2],
                normals[i * (div_lat-1) + div_lat-2]
                ])
        
        GL.triangleSet(triangulos, colors, normals = np.array(t_normals).flatten())

        


    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.


        div_long = 15
        
        vertices = []
        delta_theta = 2 * np.pi / div_long

        for i in range(div_long+1):
            theta = i * delta_theta
            x = bottomRadius * np.cos(theta)
            y = 0
            z = bottomRadius * np.sin(theta)
            vertices.append([x, y, z])

        vertices.append([0, height, 0])

        for i in range(div_long):
            v1 = vertices[i]
            v2 = vertices[i + 1]
            v3 = vertices[-1]
            GL.triangleSet([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]], colors)
        

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        div_long = 15
        div_lat = 15

        vertices = []
        delta_theta = 2 * np.pi / div_long
        delta_phi = height / div_lat

        for i in range(div_long+1):
            theta = i * delta_theta
            for j in range(div_lat+1):
                phi = j * delta_phi
                x = radius * np.cos(theta)
                y = phi - height/2
                z = radius * np.sin(theta)
                vertices.append([x, y, z])

        for i in range(div_long):
            for j in range(div_lat):
                p1 = i * (div_lat+1) + j
                p2 = p1 + 1
                p3 = (i + 1) * (div_lat+1) + j
                p4 = p3 + 1

                v1 = vertices[p1]
                v2 = vertices[p2]
                v3 = vertices[p3]
                v4 = vertices[p4]

                GL.triangleSet([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]], colors)
                GL.triangleSet([v2[0], v2[1], v2[2], v3[0], v3[1], v3[2], v4[0], v4[1], v4[2]], colors)


    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        if headlight:
            GL.directionalLight(0, [1, 1, 1], 1, [0, 0, -1])


    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        GL.directional_light = {
            "ambientIntensity": ambientIntensity,
            "color": color, 
            "intensity": intensity, 
            "direction": np.array(direction)
            }


    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        GL.supersampling.fill(0)
        GL.zBuffer.fill(-np.inf)


        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.  
        if loop:
            fraction_changed = ((epoch - GL.start_time) % cycleInterval) / cycleInterval
        else:
            fraction_changed = np.clip((epoch - GL.start_time) / cycleInterval, 0, 1)


        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.


        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        initial_key = 0
        end_key = 1
        for i in range(len(key) - 1):
            if key[i] <= set_fraction <= key[i + 1]:
                initial_key = i
                end_key = i + 1
                break

        key = np.array(key)
        keyValue = np.array(keyValue).reshape(-1, 3)

        d_key = key[end_key] - key[initial_key]
        s = (set_fraction - key[initial_key])/d_key
        s_matrix = np.array([
            s**3,
            s**2,
            s,
            1
        ])

        if key[initial_key] == 0:
            if closed:
                d0 = keyValue[initial_key+1] - keyValue[-1]
            else :
                d0 = np.array([0, 0, 0])
        else:
            d0 = keyValue[initial_key+1] - keyValue[initial_key-1]

        if key[end_key] == 1:
            if closed:
                d1 = keyValue[0] - keyValue[end_key-1]
            else :
                d1 = np.array([0, 0, 0])
        else:
            d1 = keyValue[end_key+1] - keyValue[end_key-1]

        print(f'KeyValues: {keyValue[initial_key]} {keyValue[end_key]} {d0} {d1}')
        c_matrix = np.array([
            keyValue[initial_key],
            keyValue[end_key],
            d0,
            d1
        ])

        hermite_matrix = np.array([
            [2, -2, 1, 1],
            [-3, 3, -2, -1],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ])

        value_changed = s_matrix @ hermite_matrix @ c_matrix

        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.


        initial_key = 0
        end_key = 1
        for i in range(len(key) - 1):
            if key[i] <= set_fraction <= key[i + 1]:
                initial_key = i
                end_key = i + 1
                break

        key = np.array(key)
        keyValue = np.array(keyValue).reshape(-1, 4)

        ri = keyValue[initial_key]
        rf = keyValue[end_key]
        
        s = (set_fraction - key[initial_key])/(key[end_key] - key[initial_key])

        axis_i = ri[:3]/np.linalg.norm(ri[:3])
        axis_f = rf[:3]/np.linalg.norm(rf[:3])

        angle_i = ri[3]
        angle_f = rf[3]

        if not np.allclose(axis_i, axis_f):
            axis_f = axis_i

        interpolated_angle = (1 - s) * angle_i + s * angle_f

        

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = np.concatenate([axis_f, [interpolated_angle]])

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
