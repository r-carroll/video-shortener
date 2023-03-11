const movies = ['Interstellar', 'Blade Runner', 'Ghost in the shell', 'Akira', 'Dr strange'];
let movieIndex = 0;
for(let index = 0; index <= 1000; index++) {
    movieIndex = Math.floor(Math.random()*movies.length);
}
console.log(movies[movieIndex]);