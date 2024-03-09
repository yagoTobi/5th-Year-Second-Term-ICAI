use tvmovistar;
-- Utilizamos la tabla use tvmovistarm

describe canales;
select *
from visualizaciones limit
5;
select *
from programas limit
100;

-- Paso 1: Pintate el esquema de las bases de datos: 
-- Visualizaciones 
-- Programas 
-- Canales

SELECT *
FROM programas
WHERE 
    upper(genero) =  'INFANTIL' AND
    upper(subgenero) = 'DIBUJOS ANIMADOS' AND
    upper(codigo_moral) NOT LIKE '%NO RECOMENDADA%';

select distinct(codigo_moral)
from programas;
select *
from ( 
    select
        *,
        case when upper(codigo_moral) like '%%NO RECOMENDADA%%' then 'adultos'
            when codigo_moral <> '' then 'desconocido' 
        else 'infantil' end as categoria
    from programas ) a
-- No te va a permitir usar una columna que aun no ha sido creada
where categoria = 'infantil';


select count(*)
from programas;
select distinct(genero)
from programas;

SELECT
    COUNT(*) as cuetna,
    count(distinct(codigo_moral)) as cuenta_codigos
from programas_format
group by genero;



select
    count(*) as cuenta, categoria
from ( 
        select
        *,
        case when upper(codigo_moral) like '%%NO RECOMENDADA%%' then 'adultos'
                when codigo_moral <> '' then 'desconocido' 
            else 'infantil' end as categoria
    from programas ) a
-- No te va a permitir usar una columna que aun no ha sido creada
group by categoria
= 'infantil';

select v.id_cliente,
    v.id_programa,
    p.nombre_programa
FROM
    visualizaciones v
    JOIN programas p
    on v.id_program=p.id_program;


CREATE table grtl24.yago_201802168 AS
SELECT v.id_client,
    v.id_programa,
    p.nombre_programa, 
FROM
    visualizaciones v
    JOIN PROGRAMAS P
    ON v.id_programa = p.id_programa
WHERE v.id_cliente = 0111;

-- En la tabal de 


SELECT
    date(it_inicio) as fecha,
    count(*) as num_visualizaciones,
    count(distinct(id_cliente)) as num_clienetes
FROM visualizaciones
GROUP BY date(it_inicio);



--- Practica con Pedro: 
SELECT
    date(it_inicio) as fecha,
    count(*) as num_visualizaciones,
    count(distinct(id_cliente)) as num_clientes
FROM visualizaciones
GROUP BY date(it_inicio);

-- tabla 2
SELECT
    id_programa,
    it_inicio,
    it_fin,
    (unix_timestamp(it_fin)-unix_timestamp(it_inicio))/60 as num_minutos
from visualizaciones;

-- tabla 3

SELECT
    id_programa,
    sum((unix_timestamp(it_fin)-unix_timestamp(it_inicio))/60 ) as tiempo_total
FROM visualizaciones
GROUP BY id_programa;

-- tabla 4

SELECT
    p.nombre_programa,
    v.id_programa,
    sum((unix_timestamp(v.it_fin)-unix_timestamp(v.it_inicio))/60 ) as tiempo_total
FROM visualizaciones v JOIN programas p on v.id_programa=p.id_programa
WHERE v.id_programa <> 0 AND v.id_programa <> -1
-- o NOT IN(-1, 0)
GROUP BY id_programa
ORDER BY tiempo_total desc
LIMIT 50;

--- Funcion Ventana SQL 
SELECT 
    row_number
() over
(order by tiempo_total desc) as ran, 
    nombre_programa
FROM
(SELECT
    p.nombre_programa,
    v.id_programa,
    sum((unix_timestamp(v.it_fin)-unix_timestamp(v.it_inicio))/60 ) as tiempo_total
FROM visualizaciones v JOIN programas p on v.id_programa=p.id_programa
WHERE v.id_programa <> 0 AND v.id_programa <> -1
-- o NOT IN(-1, 0)
GROUP BY id_programa
ORDER BY tiempo_total desc
LIMIT 50;)