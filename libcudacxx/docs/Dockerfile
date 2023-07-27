FROM jekyll/jekyll:4.0
COPY Gemfile /Gemfile
RUN /bin/bash -l -c "bundle install"
RUN /bin/bash -l -c "bundle add webrick"