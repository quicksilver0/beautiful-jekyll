Highcharts.chart('container', {
    chart: {
        type: 'spline'
    },
    title: {
        text: 'Most Frequent Headlines from HackerNews'
    },
    subtitle: {
        text: '2010-2015 quarter periods'
    },
    xAxis: {
        categories: ['2010q1','2010q2','2010q3','2010q4','2011q1','2011q2','2011q3','2011q4','2012q1','2012q2','2012q3','2012q4','2013q1','2013q2','2013q3','2013q4','2014q1','2014q2','2014q3','2014q4']
    },
    yAxis: {
        title: {
            text: 'Number of Posts'
        },
    },
    tooltip: {
        crosshairs: true,
        shared: false
    },
    plotOptions: {
        spline: {
        		lineWidth: .4,
            color: '#222222',
            marker: {
                radius: 1,
                lineColor: '#666666',              
            },
            states: {
                  hover: {
                      lineWidthPlus: 3
                  }
              },
              
           events: {
                    mouseOver: function () {
                        
                        this.chart.series[this.index].update({
                        	color: '#3186ad'
                        });
                    },
                    mouseOut: function () {
                        
                        this.chart.series[this.index].update({
                        	color: "#222222"
                        });                           
                    }
                }   
              
        },
        series: {
        stickyTracking: false
        }
    },
    
    annotations: [
    	{
        labelOptions: {
            backgroundColor: 'rgba(255,255,255,0.5)',
            verticalAlign: 'left',
            y: -45
        },
        labels: [{
            point: {
                xAxis: 0,
                yAxis: 0,
                x: 5,
                y: 95
            },
            text: 'Hispanic telenovellas,<br>suddenly...'
        }]
    },{
        labelOptions: {
            verticalAlign:'left',
            y: -75
        },
        labels: [{
            point: {
                xAxis: 0,
                yAxis: 0,
                x: 13,
                y: 145
            },
            text: 'Google Glass o_o<br>Big Data'
        },
        {
            point: {
                xAxis: 0,
                yAxis: 0,
                x: 19,
                y: 95
            },
            text: 'Elon Musk<br>Artificial Intelligence'
        }
        ]
    },
    ],
    
    series: [
    {
        name: 'Angry Birds',
        data: ['null', 1, 4, 14, {
            y: 45,
            marker: {
                symbol: 'url(https://i.imgur.com/rUtKIzi.png)'
            }
        }, 25, 15, 18, 20, 6, 11, 1, 3, 5, 2, 4, 5, 3, 'null', 2]
    },    
    {
        name: 'Triunfo Del Amor - Capítulo',
        data: ['null', 'null', 'null', 1, 9, 73, 'null', 'null', 'null', 1, 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null']
    },
    {
        name: 'Reina Del Sur - Capitulo',
        data: ['null', 'null', 'null', 'null', 3, 66, 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null']
    },
    {
        name: 'Real Estate',
        data: [10, 8, 12, 6, 7, 12, 38,  {
            y: 65,
            marker: {
                symbol: 'url(https://i.imgur.com/TMk4g2A.png)'
            }
        }, 17, 15, 8, 7, 3, 1, 7, 5, 4, 7, 8, 6]
    },
    {
        name: 'Steve Jobs',
        data: [18, 78, 32, 39, 70, 37, 93, {
            y: 396,
            marker: {
                symbol: 'url(https://i.imgur.com/AqTOBIU.png)'
            }
        }, 56, 49, 39, 49, 32, 29, 39, 30, 38, 20, 8, 17]

    },
    {
        name: 'New Ipad',
        data: ['null', 1, 1, 'null', 5, 2, 1, 2,  {
            y: 115,
            marker: {
                symbol: 'url(https://i.imgur.com/Kyfisxm.png)'
            }
        }, 20, 3, 2, 1, 1, 1, 1, 'null', 'null', 1, 'null']
    },
    {
        name: 'Silicon Valley',
        data: [10, 17, 11, 27, 34, 40, 41, 41, 42, 59, 45, 35, 42, 53, 45, 53, 77, 48, 59, 31]
    },
    {
        name: 'Windows Phone',
        data: [6, 1, 18, 35, 30, 14, 16, 22, 25, 36, 15, 18, 20, 10, 11, 9, 14, 7, 4, 3]
    },
    {
        name: 'Ipad Mini',
        data: ['null', 'null', 'null', 'null', 'null', 'null', 'null', 2, 4, 4, 13, 53, 7, 4, 1, 2, 'null', 'null', 'null', 3]
    },
    {
        name: 'Aaron Swartz',
        data: ['null', 'null', 'null', 1, 'null', 1, 3, 1, 1, 1, 2, 'null',  {
            y: 122,
            marker: {
                symbol: 'url(https://i.imgur.com/vmpaOS2.png)'
            }
        }, 7, 7, 1, 8, 2, 2, 'null']
    },
    {
        name: 'Big Data',
        data: [5, 2, 9, 12, 19, 17, 26, 30, 66, 73, 70, 89, 116, 124, 115, 85, 77, 64, 69, 64]
    },
    {
        name: 'Google Glass',
        data: ['null', 'null', 'null', 'null', 'null', 'null', 'null', 1, 5, 17, 14, 7, 95,{
            y: 125,
            marker: {
                symbol: 'url(https://i.imgur.com/gMBHyOx.png)'
            }
        }, 57, 55, 43, 64, 17, 16]
    },
    {
        name: 'Flappy Birds',
        data: ['null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null',  {
            y: 42,
            marker: {
                symbol: 'url(https://i.imgur.com/KMbGXzD.png)'
            }
        }, 11, 5, 2]
    },
    {
        name: 'Net Neutrality',
        data: [6, 13, 14, 12, 4, 7, 1, 4, 2, 5, 2, 'null', 1, 1, 7, 5, 33, 49, 21, 32]
    },
    {
        name: 'Artificial Intelligence',
        data: [5, 8, 4, 10, 10, 6, 7, 8, 7, 11, 11, 17, 13, 8, 9, 22, 16, 14, 23, 47 ]
    },
    {
        name: 'Elon Musk',
        data: ['null', 3, 8, 'null', 2, 2, 2, 6, 2, 6, 10, 16, 24, 22, 36, 22, 8, 32, 16,  {
            y: 42,
            marker: {
                symbol: 'url(https://i.imgur.com/95eB0xp.png)'
            }
        }]
    },]   
});
