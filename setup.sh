mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"cruz.rocha.bruno@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCoRS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml