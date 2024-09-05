Leyenda
nRows= len(lista_colores) // 12 + 1
        nCols= 12
        indice= 0
        i= 0

        for row in range(1, nRows+1):
            
            if row == nRows:
                limite= len(lista_colores) %  nCols
            else:
                limite= nCols

            leyenda+= "<tr> "
            for col in range(0, limite):
                color= lista_colores[indice]
                entidad=lista_entidades[indice]
                indice+=1
                leyenda+= "<td style='text-align:center; vertical-align:middle'> "
                leyenda+=  """ <div class= 'colores' id='rectangle' style='width:25px; \
                    height:25px; background:""" + color + "'" 
                leyenda+= """ onclick=showLegend('entidad')> </div> """
                leyenda= leyenda.replace("entidad", entidad)
                leyenda+= "</td> "
            
            leyenda+= "</tr>"

            leyenda+= "<tr>"
            for col in range(0, limite):
                leyenda+="<td>"
                leyenda+="<p style='font-size:x-small'>"
                leyenda+= lista_entidades[i]   
                i+=1
                leyenda+="</p></td>"
            leyenda+= "</tr> "    

            