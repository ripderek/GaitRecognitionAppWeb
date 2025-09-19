/*
<script>
    const canvas = document.getElementById('videoCanvas');
    const ctx = canvas.getContext('2d');
    const img = document.getElementById('videoStream');

    let personas = [];
    window.selectedId = null;

    // Dibujar cajas encima del video
    function drawBoxes() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      personas.forEach(({tid, box}) => {
        const [x1, y1, x2, y2] = box;
        ctx.strokeStyle = (tid === window.selectedId) ? 'red' : 'green';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = '16px Arial';
        //ctx.fillText(`ID ${tid}`, x1, y1 - 5);
      });

      requestAnimationFrame(drawBoxes);
    }

    drawBoxes(); // iniciar loop de dibujo

    // Actualizar detecciones cada 100ms
    setInterval(async () => {
      try {
        const res = await fetch('/get_detections');
        personas = await res.json();
      } catch (e) {
        console.error('Error al obtener detecciones:', e);
      }
    }, 100);

    // Manejar clicks sobre canvas
    canvas.addEventListener('click', function(e) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      for (let i = 0; i < personas.length; i++) {
        const {tid, box} = personas[i];
        const [x1, y1, x2, y2] = box;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
          fetch('/select_person', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ tid })
          })
          .then(res => res.json())
          .then(data => { 
            console.log('Seleccion actualizada:', data);
            window.selectedId = tid; 
          });
          break;
        }
      }
    });
  </script>
*/
