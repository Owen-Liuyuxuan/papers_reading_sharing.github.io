console.log('come here')

TESTER = document.getElementById('tester');
Plotly.plot(TESTER, [{
    x: [1, 2, 3, 4, 5],
    y: [0, 0.2, 0.4, 0.6, 1.0]
}], {
    margin: { t: 0 }
});

function get_random_y(){
    return [Math.random(), Math.random(), Math.random(), Math.random(), Math.random()]
}

TESTER.on("plotly_click", randomize)

function randomize() {
    Plotly.animate(TESTER, {
        data: [{ y: get_random_y()}],
        traces: [0],
        layout: {}
    }, {
        transition: {
            duration: 500,
            easing: 'cubic-in-out'
        },
        frame: {
            duration: 500
        }
    })
}