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
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        emissiva = colors['emissiveColor']
        emissiva = [int(emissiva[0]*255), int(emissiva[1]*255), int(emissiva[2]*255)]

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
            if area>0: # clockwise
                x0,y0,x1,y1,x2,y2 = x1,y1,x0,y0,x2,y2 # swap points
                area = -area # invert area


            min_x = math.floor(min(x0,x1,x2))
            max_x = math.ceil(max(x0,x1,x2))
            min_y = math.floor(min(y0,y1,y2))
            max_y = math.ceil(max(y0,y1,y2))


            
            for x in range(min_x, max_x+1):
                for y in range(min_y, max_y+1):
                    if inside([x0,y0], [x1,y1], [x2,y2], [x+0.5,y+0.5]) and x>=0 and y>=0 and x<GL.width and y<GL.height:
                        gpu.GPU.draw_pixel((x, y), gpu.GPU.RGB8, emissiva)
                        
    @staticmethod
    def triangleSet(point, colors):
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
        
        n_triangles = len(point) // 9
        for i in range(0, n_triangles):
            p = point[i*9:i*9+9]
            x = p[0:9:3]
            y = p[1:9:3]
            z = p[2:9:3]
            tri_mat = np.array([x, y, z, [1, 1, 1]])
            tri_mat = GL.getMatrix() @ tri_mat

            tri_mat = GL.perspective_matrix @ tri_mat
            tri_mat = tri_mat / tri_mat[3][0]
            screen_matrix = GL.to_screen_matrix(GL.width, GL.height) @ tri_mat

            screen_matrix = np.array(screen_matrix)
            GL.triangleSet2D([screen_matrix[0][0], screen_matrix[1][0], screen_matrix[0][1], screen_matrix[1][1], screen_matrix[0][2], screen_matrix[1][2]], colors)


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
        if translation:
            GL.translation = translation
            #print("translation = {0} ".format(translation), end='') # imprime no terminal
        if scale:
            GL.scale = scale
            #print("scale = {0} ".format(scale), end='') # imprime no terminal
        if rotation:
            GL.rotation = rotation
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

        for i in range(0, len(point)-6, 3):
            v1 = point[i:i+3]
            v2 = point[i+3:i+6]
            v3 = point[i+6:i+9]
            GL.triangleSet([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]], colors)



    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        
        i = 0 
        while i < len(index) - 2: 
            if index[i] == -1 or index[i + 1] == -1 or index[i + 2] == -1:
                i += 1
                continue  # Para a execução se encontrar -1 no índice
            
            # Pega as coordenadas dos vértices usando os índices da lista 'index'
            v1 = point[3 * index[i] : 3 * index[i] + 3]  # Coordenadas do primeiro vértice
            v2 = point[3 * index[i + 1] : 3 * index[i + 1] + 3]  # Coordenadas do segundo vértice
            v3 = point[3 * index[i + 2] : 3 * index[i + 2] + 3]  # Coordenadas do terceiro vértice
            
            GL.triangleSet([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]], colors)
            
            # Avança para o próximo conjunto de vértices
            i += 1

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                    texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""

        # Processamento das faces baseado em coordIndex
        faces = []
        vertices = []

        # Loop para construir as faces a partir de coordIndex
        for i in coordIndex:
            if i == -1 and len(vertices)>0:
                faces.append(vertices)
                vertices = []  # Reseta a lista de vértices para a próxima face
            else:
                vertices.append(i)  # Adiciona vértices à face atual

        # Criação de strips a partir das faces
        strips = []
        for f in faces:
            if len(f) < 3:
                continue  # Ignora faces com menos de 3 vértices (não podem formar triângulos)

            # Gera strips para a face
            for i in range(1, len(f) - 1):
                strips.extend([f[0], f[i], f[i + 1], -1])  # Adiciona um strip com final -1

        # Chama a função para desenhar os strips de triângulos
        GL.indexedTriangleStripSet(coord, strips, colors)


    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
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
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
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

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
